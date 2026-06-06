"""Mirror-invariance benchmark helpers for swing analysis results.

The script compares two completed analyses, usually an original video and a
deterministically generated horizontal mirror, and reports metric deltas against
explicit tolerances.

Examples:
    python -m scripts.mirror_invariance_benchmark hflip \
        --input swing.mov --output swing_hflip.mp4

    python -m scripts.mirror_invariance_benchmark compare \
        --baseline-json original_metrics.json --mirror-json mirror_metrics.json \
        --output reports/mirror_benchmark.md

    python -m scripts.mirror_invariance_benchmark compare-db \
        --baseline-analysis-id d369491e-e453-4681-b32b-c5b37e0bc9a9 \
        --mirror-analysis-id 1c85ceb2-d37c-4350-b3ad-b16007a3fce7 \
        --format json --fail-on-regression
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricSpec:
    name: str
    tolerance: float
    unit: str
    description: str


@dataclass(frozen=True)
class MetricComparison:
    name: str
    baseline_value: float | None
    mirror_value: float | None
    delta: float | None
    tolerance: float
    unit: str
    passed: bool
    description: str


DEFAULT_METRIC_SPECS: tuple[MetricSpec, ...] = (
    MetricSpec("bat_speed", 10.0, "km/h", "Impact-zone bat speed"),
    MetricSpec("attack_angle", 8.0, "deg", "2-frame impact attack angle"),
    MetricSpec("hand_path_efficiency", 0.15, "ratio", "Lead-hand path efficiency"),
    MetricSpec("stride_length_cm", 5.0, "cm", "Stride length"),
    MetricSpec("impact_frame", 3.0, "frames", "Selected impact frame"),
    MetricSpec("phase_ms.stance", 120.0, "ms", "Stance phase duration"),
    MetricSpec("phase_ms.load", 120.0, "ms", "Load phase duration"),
    MetricSpec("phase_ms.stride", 120.0, "ms", "Stride phase duration"),
    MetricSpec("phase_ms.rotation", 120.0, "ms", "Rotation phase duration"),
    MetricSpec("phase_ms.impact", 120.0, "ms", "Impact phase duration"),
    MetricSpec("phase_ms.follow_through", 120.0, "ms", "Follow-through duration"),
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def load_analysis_from_db(analysis_id: str) -> dict[str, Any]:
    from app.db.models import AnalysisResultTable
    from app.db.session import sync_session_factory

    session = sync_session_factory()
    try:
        result = (
            session.query(AnalysisResultTable)
            .filter(AnalysisResultTable.analysis_id == uuid.UUID(analysis_id))
            .first()
        )
        if result is None:
            raise ValueError(f"Analysis result not found: {analysis_id}")
        return {
            "analysis_id": str(result.analysis_id),
            "biomechanics": result.biomechanics_data or {},
            "swing_phases": result.swing_phases_data or {},
            "processing_time_seconds": result.processing_time_seconds,
        }
    finally:
        session.close()


def as_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    return None


def nested_number(data: dict[str, Any], *path: str) -> float | None:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return as_number(current)


def first_number(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def extract_metric_values(payload: dict[str, Any]) -> dict[str, float]:
    """Extract benchmark metrics from report, metrics endpoint, or DB-shaped JSON."""
    biomechanics = payload.get("biomechanics")
    if biomechanics is None:
        biomechanics = payload.get("biomechanics_data")
    if not isinstance(biomechanics, dict):
        biomechanics = payload

    values: dict[str, float] = {}

    scalar_extractors = {
        "bat_speed": first_number(
            nested_number(biomechanics, "bat_speed", "speed_kmh"),
            nested_number(biomechanics, "bat_speed", "measured_value"),
            nested_number(biomechanics, "bat_speed", "value"),
        ),
        "attack_angle": first_number(
            nested_number(biomechanics, "attack_angle", "angle_degrees"),
            nested_number(biomechanics, "attack_angle", "measured_value"),
            nested_number(biomechanics, "attack_angle", "value"),
        ),
        "hand_path_efficiency": nested_number(biomechanics, "hand_path_efficiency"),
        "stride_length_cm": nested_number(biomechanics, "stride_length_cm"),
        "impact_frame": first_number(
            nested_number(biomechanics, "impact_frame"),
            nested_number(biomechanics, "attack_angle", "impact_frame"),
        ),
    }
    for name, value in scalar_extractors.items():
        if value is not None:
            values[name] = value

    for phase, duration in extract_phase_durations(payload).items():
        values[f"phase_ms.{phase}"] = duration

    return values


def extract_phase_durations(payload: dict[str, Any]) -> dict[str, float]:
    phases_payload: Any = (
        payload.get("swing_phases")
        or payload.get("swing_phases_data")
        or payload.get("phases")
        or {}
    )

    durations: dict[str, float] = {}
    if isinstance(phases_payload, dict):
        raw_durations = phases_payload.get("phase_durations_ms", {})
        if isinstance(raw_durations, dict):
            for phase, value in raw_durations.items():
                number = as_number(value)
                if number is not None:
                    durations[normalize_phase_name(str(phase))] = number
        return durations

    if isinstance(phases_payload, list):
        for item in phases_payload:
            if not isinstance(item, dict):
                continue
            phase = item.get("phase")
            duration = as_number(item.get("duration_ms"))
            if phase and duration is not None:
                durations[normalize_phase_name(str(phase))] = duration

    return durations


def normalize_phase_name(phase: str) -> str:
    return phase.lower().replace("-", "_")


def compare_metric_values(
    baseline: dict[str, float],
    mirror: dict[str, float],
    specs: tuple[MetricSpec, ...] = DEFAULT_METRIC_SPECS,
) -> list[MetricComparison]:
    comparisons: list[MetricComparison] = []
    for spec in specs:
        baseline_value = baseline.get(spec.name)
        mirror_value = mirror.get(spec.name)
        if baseline_value is None or mirror_value is None:
            comparisons.append(
                MetricComparison(
                    name=spec.name,
                    baseline_value=baseline_value,
                    mirror_value=mirror_value,
                    delta=None,
                    tolerance=spec.tolerance,
                    unit=spec.unit,
                    passed=False,
                    description=spec.description,
                )
            )
            continue

        delta = abs(baseline_value - mirror_value)
        comparisons.append(
            MetricComparison(
                name=spec.name,
                baseline_value=baseline_value,
                mirror_value=mirror_value,
                delta=delta,
                tolerance=spec.tolerance,
                unit=spec.unit,
                passed=delta <= spec.tolerance,
                description=spec.description,
            )
        )
    return comparisons


def render_markdown(comparisons: list[MetricComparison]) -> str:
    lines = [
        "# Mirror Invariance Benchmark",
        "",
        "| Metric | Original | Mirror | Delta | Tolerance | Result | Notes |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in comparisons:
        result = "PASS" if item.passed else "FAIL"
        lines.append(
            "| {name} | {base} | {mirror} | {delta} | {tol} {unit} | {result} | {desc} |".format(
                name=item.name,
                base=format_value(item.baseline_value),
                mirror=format_value(item.mirror_value),
                delta=format_value(item.delta),
                tol=format_value(item.tolerance),
                unit=item.unit,
                result=result,
                desc=item.description,
            )
        )
    return "\n".join(lines) + "\n"


def render_json(comparisons: list[MetricComparison]) -> str:
    passed = all(item.passed for item in comparisons)
    payload = {
        "passed": passed,
        "comparisons": [
            {
                "metric": item.name,
                "baseline_value": item.baseline_value,
                "mirror_value": item.mirror_value,
                "delta": item.delta,
                "tolerance": item.tolerance,
                "unit": item.unit,
                "passed": item.passed,
                "description": item.description,
            }
            for item in comparisons
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def format_value(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def write_or_print(text: str, output: Path | None) -> None:
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def run_hflip(input_path: Path, output_path: Path, overwrite: bool) -> None:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-vf",
        "hflip,fps=30,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-an",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    hflip = subparsers.add_parser("hflip", help="Generate a deterministic hflip video")
    hflip.add_argument("--input", type=Path, required=True)
    hflip.add_argument("--output", type=Path, required=True)
    hflip.add_argument("--overwrite", action="store_true")

    compare = subparsers.add_parser("compare", help="Compare two JSON result payloads")
    compare.add_argument("--baseline-json", type=Path, required=True)
    compare.add_argument("--mirror-json", type=Path, required=True)
    add_compare_output_args(compare)

    compare_db = subparsers.add_parser("compare-db", help="Compare two DB analysis results")
    compare_db.add_argument("--baseline-analysis-id", required=True)
    compare_db.add_argument("--mirror-analysis-id", required=True)
    add_compare_output_args(compare_db)

    return parser


def add_compare_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero when any metric is missing or outside tolerance",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "hflip":
        run_hflip(args.input, args.output, args.overwrite)
        return 0

    try:
        if args.command == "compare":
            baseline_payload = load_json(args.baseline_json)
            mirror_payload = load_json(args.mirror_json)
        elif args.command == "compare-db":
            baseline_payload = load_analysis_from_db(args.baseline_analysis_id)
            mirror_payload = load_analysis_from_db(args.mirror_analysis_id)
        else:
            raise ValueError(f"Unsupported command: {args.command}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    comparisons = compare_metric_values(
        extract_metric_values(baseline_payload),
        extract_metric_values(mirror_payload),
    )
    rendered = render_json(comparisons) if args.format == "json" else render_markdown(comparisons)
    write_or_print(rendered, args.output)

    if args.fail_on_regression and not all(item.passed for item in comparisons):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
