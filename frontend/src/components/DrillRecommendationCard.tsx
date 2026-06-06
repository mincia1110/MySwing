/**
 * Renders a single drill recommendation (Requirement 8.4).
 */
import type { DrillRecommendationResponse } from "../types/analysis";
import "./DrillRecommendationCard.css";

export interface DrillRecommendationCardProps {
  drill: DrillRecommendationResponse;
}

const DIRECTION_LABELS: Record<NonNullable<DrillRecommendationResponse["direction"]>, string> = {
  below: "기준 미달",
  above: "기준 초과",
  generic: "맞춤 안내",
};

export function DrillRecommendationCard({ drill }: DrillRecommendationCardProps) {
  const directionLabel =
    drill.direction && DIRECTION_LABELS[drill.direction]
      ? DIRECTION_LABELS[drill.direction]
      : null;
  return (
    <article
      className="drill-card"
      data-testid={`drill-card-${drill.drill_name}`}
      aria-label={`드릴 추천: ${drill.drill_name}`}
    >
      <h4 className="drill-card__name" data-testid="drill-card-name">
        {drill.drill_name}
      </h4>
      <span className="drill-card__target" data-testid="drill-card-target">
        대상: {drill.target_metric}
      </span>
      {directionLabel ? (
        <span
          className={`drill-card__direction drill-card__direction--${drill.direction}`}
          data-testid="drill-card-direction"
        >
          방향: {directionLabel}
        </span>
      ) : null}
      <p className="drill-card__description" data-testid="drill-card-description">
        {drill.description}
      </p>
    </article>
  );
}

export default DrillRecommendationCard;
