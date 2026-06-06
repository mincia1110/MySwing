/**
 * Top-3 ranked improvement areas list (Requirement 7.6 / 8.4).
 *
 * Items are expected to be supplied pre-ranked (rank=1..3) by the backend;
 * this component sorts by rank as a defensive measure.
 */
import type { ImprovementAreaResponse } from "../types/analysis";
import "./ImprovementAreasList.css";

export interface ImprovementAreasListProps {
  improvements: ImprovementAreaResponse[];
  title?: string;
}

export function ImprovementAreasList({
  improvements,
  title = "개선이 필요한 영역 (상위 3개)",
}: ImprovementAreasListProps) {
  const sorted = [...improvements].sort((a, b) => a.rank - b.rank);

  return (
    <section
      className="improvements"
      aria-label="개선이 필요한 영역"
      data-testid="improvements"
    >
      <h3 className="improvements__title">{title}</h3>
      {sorted.length === 0 ? (
        <p className="improvements__empty" data-testid="improvements-empty">
          개선이 필요한 영역이 식별되지 않았습니다.
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
              <span className="improvements__rank" aria-label={`순위 ${imp.rank}`}>
                {imp.rank}
              </span>
              <span className="improvements__metric">
                <span className="improvements__metric-name">
                  {imp.metric_name}
                </span>
                <span className="improvements__metric-detail">
                  현재값 {imp.current_value.toFixed(1)} / 목표{" "}
                  {imp.target_range_min.toFixed(1)} -{" "}
                  {imp.target_range_max.toFixed(1)}
                </span>
              </span>
              <span
                className="improvements__deviation"
                data-testid={`improvements-deviation-${imp.metric_name}`}
              >
                편차 {imp.deviation_percent.toFixed(1)}%
              </span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

export default ImprovementAreasList;
