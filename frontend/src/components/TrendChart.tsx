/**
 * Simple SVG-based time-series chart for metric trends
 * (Requirements 8.7, 8.8).
 *
 * - When `totalRecordings` < `MIN_RECORDINGS_FOR_TREND` (2), the chart is
 *   replaced with a message indicating the minimum number of recordings
 *   required for trend analysis.
 * - Otherwise renders a polyline of values per metric over time, capped to
 *   the most recent `MAX_TREND_RECORDINGS` (30) points.
 */
import {
  MAX_TREND_RECORDINGS,
  MIN_RECORDINGS_FOR_TREND,
  type MetricDataPointResponse,
  type TrendDataResponse,
} from "../types/analysis";
import "./TrendChart.css";

export interface TrendChartProps {
  trendData: TrendDataResponse;
  /**
   * Optional list of metric names to render. Defaults to all available
   * metrics in `trendData.metrics_history`.
   */
  metricNames?: string[];
  width?: number;
  height?: number;
  title?: string;
}

interface ChartGeometry {
  width: number;
  height: number;
  padding: number;
  innerWidth: number;
  innerHeight: number;
}

function computeGeometry(width: number, height: number): ChartGeometry {
  const padding = 32;
  return {
    width,
    height,
    padding,
    innerWidth: Math.max(0, width - padding * 2),
    innerHeight: Math.max(0, height - padding * 2),
  };
}

function buildPath(
  points: MetricDataPointResponse[],
  geom: ChartGeometry,
  minVal: number,
  maxVal: number,
): { path: string; coords: Array<{ x: number; y: number }> } {
  const range = maxVal - minVal || 1;
  const stepX = points.length > 1 ? geom.innerWidth / (points.length - 1) : 0;
  const coords = points.map((p, i) => ({
    x: geom.padding + i * stepX,
    y:
      geom.padding +
      geom.innerHeight -
      ((p.value - minVal) / range) * geom.innerHeight,
  }));
  const path = coords
    .map((c, i) => `${i === 0 ? "M" : "L"}${c.x.toFixed(1)},${c.y.toFixed(1)}`)
    .join(" ");
  return { path, coords };
}

function pickRecentPoints(
  points: MetricDataPointResponse[],
): MetricDataPointResponse[] {
  // Backend may return points in any order; sort by recorded_at ascending
  // and clamp to the most recent MAX_TREND_RECORDINGS.
  const sorted = [...points].sort(
    (a, b) =>
      new Date(a.recorded_at).getTime() - new Date(b.recorded_at).getTime(),
  );
  if (sorted.length <= MAX_TREND_RECORDINGS) return sorted;
  return sorted.slice(sorted.length - MAX_TREND_RECORDINGS);
}

function MinRecordingsMessage({
  total,
  message,
}: {
  total: number;
  message?: string | null;
}) {
  const text =
    message ??
    `트렌드 분석을 위해서는 최소 ${MIN_RECORDINGS_FOR_TREND}회 이상의 분석 기록이 필요합니다 (현재 ${total}회).`;
  return (
    <p className="trend-chart__message" data-testid="trend-chart-message">
      {text}
    </p>
  );
}

export function TrendChart({
  trendData,
  metricNames,
  width = 600,
  height = 220,
  title = "메트릭 추이",
}: TrendChartProps) {
  const total = trendData.total_recordings;

  if (total < MIN_RECORDINGS_FOR_TREND) {
    return (
      <section
        className="trend-chart"
        aria-label="메트릭 추이"
        data-testid="trend-chart"
        data-state="insufficient"
      >
        <h3 className="trend-chart__title">{title}</h3>
        <MinRecordingsMessage total={total} message={trendData.message} />
      </section>
    );
  }

  const allMetricNames = metricNames ?? Object.keys(trendData.metrics_history);
  const geom = computeGeometry(width, height);

  if (allMetricNames.length === 0) {
    return (
      <section
        className="trend-chart"
        aria-label="메트릭 추이"
        data-testid="trend-chart"
        data-state="empty"
      >
        <h3 className="trend-chart__title">{title}</h3>
        <p className="trend-chart__message" data-testid="trend-chart-empty">
          트렌드 데이터가 없습니다.
        </p>
      </section>
    );
  }

  return (
    <section
      className="trend-chart"
      aria-label="메트릭 추이"
      data-testid="trend-chart"
      data-state="ready"
      data-total-recordings={total}
    >
      <h3 className="trend-chart__title">{title}</h3>
      {allMetricNames.map((metricName) => {
        const rawPoints = trendData.metrics_history[metricName] ?? [];
        const points = pickRecentPoints(rawPoints);
        if (points.length === 0) return null;

        const values = points.map((p) => p.value);
        const minVal = Math.min(...values);
        const maxVal = Math.max(...values);
        const { path, coords } = buildPath(points, geom, minVal, maxVal);

        return (
          <div
            key={metricName}
            data-testid={`trend-chart-metric-${metricName}`}
            data-points={points.length}
          >
            <p className="trend-chart__metric-name">{metricName}</p>
            <svg
              className="trend-chart__svg"
              viewBox={`0 0 ${geom.width} ${geom.height}`}
              role="img"
              aria-label={`${metricName} 추이 차트`}
              preserveAspectRatio="none"
            >
              <line
                className="trend-chart__axis"
                x1={geom.padding}
                y1={geom.padding + geom.innerHeight}
                x2={geom.padding + geom.innerWidth}
                y2={geom.padding + geom.innerHeight}
              />
              <line
                className="trend-chart__axis"
                x1={geom.padding}
                y1={geom.padding}
                x2={geom.padding}
                y2={geom.padding + geom.innerHeight}
              />
              <text
                className="trend-chart__label"
                x={geom.padding}
                y={geom.padding - 6}
              >
                {maxVal.toFixed(1)}
              </text>
              <text
                className="trend-chart__label"
                x={geom.padding}
                y={geom.padding + geom.innerHeight + 14}
              >
                {minVal.toFixed(1)}
              </text>
              <path className="trend-chart__path" d={path} />
              {coords.map((c, idx) => (
                <circle
                  key={idx}
                  className={`trend-chart__point trend-chart__point--${points[idx].rating}`}
                  cx={c.x}
                  cy={c.y}
                  r={3}
                  data-testid={`trend-chart-point-${metricName}-${idx}`}
                />
              ))}
            </svg>
          </div>
        );
      })}
    </section>
  );
}

export default TrendChart;
