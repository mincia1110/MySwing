import json

from scripts.mirror_invariance_benchmark import (
    compare_metric_values,
    extract_metric_values,
    main,
    render_markdown,
)


def _payload(
    bat_speed=92.9,
    attack_angle=17.0,
    hand_path_efficiency=0.54,
    stride_length_cm=116.0,
    impact_frame=42,
):
    return {
        "biomechanics": {
            "bat_speed": {"speed_kmh": bat_speed, "precision": 1.0},
            "attack_angle": {
                "angle_degrees": attack_angle,
                "precision": 0.5,
                "impact_frame": impact_frame,
            },
            "hand_path_efficiency": hand_path_efficiency,
            "stride_length_cm": stride_length_cm,
        },
        "swing_phases": {
            "phase_durations_ms": {
                "stance": 300.0,
                "load": 260.0,
                "stride": 240.0,
                "rotation": 180.0,
                "impact": 70.0,
                "follow_through": 400.0,
            }
        },
    }


def test_extract_metric_values_supports_report_shaped_payload():
    values = extract_metric_values(_payload())

    assert values["bat_speed"] == 92.9
    assert values["attack_angle"] == 17.0
    assert values["hand_path_efficiency"] == 0.54
    assert values["stride_length_cm"] == 116.0
    assert values["impact_frame"] == 42.0
    assert values["phase_ms.load"] == 260.0
    assert values["phase_ms.follow_through"] == 400.0


def test_compare_metric_values_marks_within_tolerance_as_passed():
    baseline = extract_metric_values(_payload(bat_speed=92.0, impact_frame=40))
    mirror = extract_metric_values(_payload(bat_speed=99.0, impact_frame=43))

    comparisons = compare_metric_values(baseline, mirror)

    assert all(item.passed for item in comparisons)


def test_compare_metric_values_marks_out_of_tolerance_as_failed():
    baseline = extract_metric_values(_payload(attack_angle=55.0))
    mirror = extract_metric_values(_payload(attack_angle=18.0))

    comparisons = compare_metric_values(baseline, mirror)
    attack_angle = next(item for item in comparisons if item.name == "attack_angle")

    assert attack_angle.delta == 37.0
    assert attack_angle.passed is False


def test_render_markdown_includes_tolerance_table():
    comparisons = compare_metric_values(
        extract_metric_values(_payload()),
        extract_metric_values(_payload()),
    )

    markdown = render_markdown(comparisons)

    assert "| Metric | Original | Mirror | Delta | Tolerance | Result | Notes |" in markdown
    assert "| bat_speed | 92.9 | 92.9 | 0 | 10 km/h | PASS |" in markdown


def test_compare_cli_writes_json_and_returns_failure_for_regression(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    mirror_path = tmp_path / "mirror.json"
    output_path = tmp_path / "result.json"
    baseline_path.write_text(json.dumps(_payload(attack_angle=55.0)), encoding="utf-8")
    mirror_path.write_text(json.dumps(_payload(attack_angle=18.0)), encoding="utf-8")

    code = main(
        [
            "compare",
            "--baseline-json",
            str(baseline_path),
            "--mirror-json",
            str(mirror_path),
            "--format",
            "json",
            "--output",
            str(output_path),
            "--fail-on-regression",
        ]
    )

    assert code == 1
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["passed"] is False
