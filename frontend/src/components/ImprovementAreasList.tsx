/**
 * Top-3 ranked improvement areas list (Requirement 7.6 / 8.4).
 *
 * Items are expected to be supplied pre-ranked (rank=1..3) by the backend;
 * this component sorts by rank as a defensive measure.
 */
import type { ImprovementAreaResponse } from "../types/analysis";
import { useTranslation } from "../i18n";
import "./ImprovementAreasList.css";

export interface ImprovementAreasListProps {
  improvements: ImprovementAreaResponse[];
  title?: string;
}

export function ImprovementAreasList({
  improvements,
  title,
}: ImprovementAreasListProps) {
  const { t } = useTranslation();
  const sorted = [...improvements].sort((a, b) => a.rank - b.rank);
  const resolvedTitle = title ?? t("improvements.title");

  return (
    <section
      className="improvements"
      aria-label={t("improvements.aria")}
      data-testid="improvements"
    >
      <h3 className="improvements__title">{resolvedTitle}</h3>
      {sorted.length === 0 ? (
        <p className="improvements__empty" data-testid="improvements-empty">
          {t("improvements.empty")}
        </p>
      ) : (
        <ol className="improvements__list">
          {sorted.map((imp) => (
            <li
              key={imp.metric_name}
              className="improvements__item"
              data-testid={`improvements-item-${imp.metric_name}`}
              data-rank={imp.rank}
            >
              <span
                className="improvements__rank"
                aria-label={t("improvements.rank", { rank: imp.rank })}
              >
                {imp.rank}
              </span>
              <span className="improvements__metric">
                <span className="improvements__metric-name">
                  {imp.metric_name}
                </span>
                <span className="improvements__metric-detail">
                  {t("improvements.currentTarget", {
                    current: imp.current_value.toFixed(1),
                    min: imp.target_range_min.toFixed(1),
                    max: imp.target_range_max.toFixed(1),
                  })}
                </span>
              </span>
              <span
                className="improvements__deviation"
                data-testid={`improvements-deviation-${imp.metric_name}`}
              >
                {t("improvements.deviation", {
                  value: imp.deviation_percent.toFixed(1),
                })}
              </span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

export default ImprovementAreasList;
