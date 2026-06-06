/**
 * Renders a single drill recommendation (Requirement 8.4).
 */
import type { DrillRecommendationResponse } from "../types/analysis";
import { useTranslation } from "../i18n";
import "./DrillRecommendationCard.css";

export interface DrillRecommendationCardProps {
  drill: DrillRecommendationResponse;
}

export function DrillRecommendationCard({ drill }: DrillRecommendationCardProps) {
  const { t } = useTranslation();
  const directionLabel =
    drill.direction
      ? t(`drill.directions.${drill.direction}`)
      : null;
  return (
    <article
      className="drill-card"
      data-testid={`drill-card-${drill.drill_name}`}
      aria-label={t("drill.aria", { name: drill.drill_name })}
    >
      <h4 className="drill-card__name" data-testid="drill-card-name">
        {drill.drill_name}
      </h4>
      <span className="drill-card__target" data-testid="drill-card-target">
        {t("drill.target", { metric: drill.target_metric })}
      </span>
      {directionLabel ? (
        <span
          className={`drill-card__direction drill-card__direction--${drill.direction}`}
          data-testid="drill-card-direction"
        >
          {t("drill.direction", { direction: directionLabel })}
        </span>
      ) : null}
      <p className="drill-card__description" data-testid="drill-card-description">
        {drill.description}
      </p>
    </article>
  );
}

export default DrillRecommendationCard;
