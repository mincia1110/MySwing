import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DrillRecommendationCard } from "./DrillRecommendationCard";

describe("DrillRecommendationCard", () => {
  it("renders drill name, target metric, and description", () => {
    render(
      <DrillRecommendationCard
        drill={{
          drill_name: "티 타격 - 어퍼 컷",
          target_metric: "attack_angle",
          description:
            "티에서 발사각을 5-15도 범위로 일관되게 만들기 위한 드릴",
          direction: "below",
        }}
      />,
    );

    expect(screen.getByTestId("drill-card-name")).toHaveTextContent(
      "티 타격 - 어퍼 컷",
    );
    expect(screen.getByTestId("drill-card-target")).toHaveTextContent(
      "attack_angle",
    );
    expect(screen.getByTestId("drill-card-description")).toHaveTextContent(
      "티에서 발사각을 5-15도",
    );
    expect(screen.getByTestId("drill-card-direction")).toHaveTextContent(
      "기준 미달",
    );
  });

  it("falls back to no direction label when direction is missing", () => {
    render(
      <DrillRecommendationCard
        drill={{
          drill_name: "일반 안내",
          target_metric: "stride_length_cm",
          description: "맞춤 훈련 설계가 필요합니다.",
        }}
      />,
    );
    expect(screen.queryByTestId("drill-card-direction")).toBeNull();
  });
});
