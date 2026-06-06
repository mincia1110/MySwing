/**
 * Side-by-side comparison of swing phase durations: user vs professional
 * reference (Requirement 8.5).
 *
 * The component is data-driven — the parent supplies the user's swing
 * phases (from the report) and an array of reference phase durations.
 * Default professional reference durations (in milliseconds) are used when
 * the parent does not specify them, matching the modern hitting model
 * documented in the design.
 */
import type { SwingPhaseResponse } from "../types/analysis";
import { useTranslation } from "../i18n";
import "./ComparisonView.css";

export interface ReferencePhase {
  phase: string;
  duration_ms: number;
}

export interface ComparisonViewProps {
  userPhases: SwingPhaseResponse[];
  referencePhases?: ReferencePhase[];
  title?: string;
}

/**
 * Default professional reference durations (ms) per phase used when the
 * caller does not supply a custom reference set.
 */
export const DEFAULT_REFERENCE_PHASES: ReferencePhase[] = [
  { phase: "stance", duration_ms: 200 },
  { phase: "load", duration_ms: 300 },
  { phase: "stride", duration_ms: 200 },
  { phase: "rotation", duration_ms: 150 },
  { phase: "impact", duration_ms: 50 },
  { phase: "follow_through", duration_ms: 250 },
];

interface ComparisonRow {
  phase: string;
  user_ms: number | null;
  reference_ms: number | null;
}

function buildRows(
  userPhases: SwingPhaseResponse[],
  referencePhases: ReferencePhase[],
): ComparisonRow[] {
  const userMap = new Map<string, number>();
  for (const p of userPhases) {
    userMap.set(p.phase.toLowerCase(), p.duration_ms);
  }
  const refMap = new Map<string, number>();
  for (const p of referencePhases) {
    refMap.set(p.phase.toLowerCase(), p.duration_ms);
  }

  const phases = new Set<string>([...userMap.keys(), ...refMap.keys()]);
  // Preserve reference order if available; append user-only phases at end.
  const orderedPhases: string[] = [];
  for (const ref of referencePhases) {
    const key = ref.phase.toLowerCase();
    if (phases.has(key) && !orderedPhases.includes(key)) {
      orderedPhases.push(key);
    }
  }
  for (const u of userPhases) {
    const key = u.phase.toLowerCase();
    if (phases.has(key) && !orderedPhases.includes(key)) {
      orderedPhases.push(key);
    }
  }

  return orderedPhases.map((phase) => ({
    phase,
    user_ms: userMap.has(phase) ? (userMap.get(phase) ?? null) : null,
    reference_ms: refMap.has(phase) ? (refMap.get(phase) ?? null) : null,
  }));
}

export function ComparisonView({
  userPhases,
  referencePhases = DEFAULT_REFERENCE_PHASES,
  title,
}: ComparisonViewProps) {
  const { t } = useTranslation();
  const rows = buildRows(userPhases, referencePhases);
  const resolvedTitle = title ?? t("comparison.title");
  const maxMs = Math.max(
    1,
    ...rows.flatMap((r) => [r.user_ms ?? 0, r.reference_ms ?? 0]),
  );

  return (
    <section
      className="comparison-view"
      aria-label={t("comparison.aria")}
      data-testid="comparison-view"
    >
      <h3 className="comparison-view__title">{resolvedTitle}</h3>
      <div className="comparison-view__legend" aria-hidden="true">
        <span>
          <span className="comparison-view__legend-swatch comparison-view__legend-swatch--user" />
          {t("comparison.user")}
        </span>
        <span>
          <span className="comparison-view__legend-swatch comparison-view__legend-swatch--reference" />
          {t("comparison.reference")}
        </span>
      </div>
      {rows.length === 0 ? (
        <p className="comparison-view__empty" data-testid="comparison-view-empty">
          {t("comparison.empty")}
        </p>
      ) : (
        <table className="comparison-view__table">
          <thead>
            <tr>
              <th scope="col">{t("comparison.phase")}</th>
              <th scope="col">{t("comparison.userMs")}</th>
              <th scope="col">{t("comparison.referenceMs")}</th>
              <th scope="col" className="comparison-view__bar-cell">
                {t("comparison.compare")}
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.phase}
                data-testid={`comparison-row-${row.phase}`}
              >
                <th scope="row">{row.phase}</th>
                <td data-testid={`comparison-user-${row.phase}`}>
                  {row.user_ms === null ? "-" : row.user_ms.toFixed(0)}
                </td>
                <td data-testid={`comparison-reference-${row.phase}`}>
                  {row.reference_ms === null ? "-" : row.reference_ms.toFixed(0)}
                </td>
                <td className="comparison-view__bar-cell">
                  <div className="comparison-view__bars">
                    <div
                      className="comparison-view__bar"
                      aria-label={t("comparison.userDuration", { phase: row.phase })}
                    >
                      <div
                        className="comparison-view__bar-fill comparison-view__bar-fill--user"
                        style={{
                          width: `${
                            row.user_ms === null
                              ? 0
                              : Math.min(100, (row.user_ms / maxMs) * 100)
                          }%`,
                        }}
                      />
                    </div>
                    <div
                      className="comparison-view__bar"
                      aria-label={t("comparison.referenceDuration", { phase: row.phase })}
                    >
                      <div
                        className="comparison-view__bar-fill comparison-view__bar-fill--reference"
                        style={{
                          width: `${
                            row.reference_ms === null
                              ? 0
                              : Math.min(100, (row.reference_ms / maxMs) * 100)
                          }%`,
                        }}
                      />
                    </div>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

export default ComparisonView;
