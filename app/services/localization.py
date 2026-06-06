"""Small localization helpers for API-facing report content."""

from typing import Literal

from app.schemas.analysis import DrillRecommendationResponse

Locale = Literal["ko", "en"]


def normalize_locale(locale: str | None) -> Locale:
    """Normalize an incoming locale value to a supported API locale."""
    return "en" if (locale or "").lower().startswith("en") else "ko"


_DRILL_TRANSLATIONS_EN: dict[str, tuple[str, str]] = {
    "오버로드 배트 티 타격": (
        "Overload Bat Tee Drill",
        "Use a heavier bat for repeated tee swings to build impact bat speed through overload training.",
    ),
    "로테이션 미디신볼 던지기": (
        "Rotational Medicine Ball Throw",
        "Throw a medicine ball with rotational intent to develop core rotation power.",
    ),
    "저항 밴드 스윙": (
        "Resistance Band Swing",
        "Swing with a resistance band attached to strengthen explosive hip and trunk rotation.",
    ),
    "컨트롤 스윙 드릴": (
        "Controlled Swing Drill",
        "Prioritize bat control and contact quality instead of maximum speed to stabilize excessive bat speed.",
    ),
    "소프트 토스 정확도 드릴": (
        "Soft Toss Accuracy Drill",
        "Repeat soft toss work while targeting a defined zone to build controlled bat speed.",
    ),
    "로우 티 드릴": (
        "Low Tee Drill",
        "Set the tee around knee height and practice consistent contact with a positive upward swing path.",
    ),
    "상향 스윙 패스 드릴": (
        "Upward Swing Path Drill",
        "Repeat a bat path that rises slightly through the ball to better match the incoming pitch plane.",
    ),
    "다단계 티 높이 드릴": (
        "Multi-Height Tee Drill",
        "Alternate tee heights to build a consistent attack angle across the strike zone.",
    ),
    "레벨 스윙 드릴": (
        "Level Swing Drill",
        "Keep the bat path flatter through the zone to reduce an excessive uppercut.",
    ),
    "탑 핸드 컨트롤 드릴": (
        "Top Hand Control Drill",
        "Strengthen top-hand control to flatten the swing plane and reduce weak fly-ball contact.",
    ),
    "하이 티 라인드라이브": (
        "High Tee Line Drive Drill",
        "Set the tee around chest height and work on line-drive contact instead of lifting the ball.",
    ),
    "힙 리드 드릴": (
        "Hip Lead Drill",
        "Start the swing with pelvic rotation while keeping the shoulders back.",
    ),
    "힙-숄더 분리 스트레칭 루틴": (
        "Hip-Shoulder Separation Mobility Routine",
        "Use a daily dynamic mobility routine to increase hip-shoulder separation.",
    ),
    "타월 드릴": (
        "Towel Connection Drill",
        "Wrap a towel across the chest to feel whether proper separation is maintained during rotation.",
    ),
    "커넥션 드릴": (
        "Connection Drill",
        "Maintain upper- and lower-body connection to reduce excessive separation.",
    ),
    "동기화 회전 드릴": (
        "Synchronized Rotation Drill",
        "Repeat smooth rotations where the hips and shoulders rotate together without over-separating.",
    ),
    "다이렉트 패스 티 드릴": (
        "Direct Path Tee Drill",
        "Focus on sending the hands directly to the ball while minimizing extra movement.",
    ),
    "숏 배트 드릴": (
        "Short Bat Drill",
        "Swing a shorter bat to encourage a compact and efficient hand path.",
    ),
    "커넥션 볼 드릴": (
        "Connection Ball Drill",
        "Hold a ball between the arm and torso to maintain connection throughout the swing.",
    ),
    "팔 뻗기 익스텐션 드릴": (
        "Extension Drill",
        "Extend the arms fully after contact so the bat travels through the ball longer.",
    ),
    "스트라이드 확장 드릴": (
        "Stride Extension Drill",
        "Practice landing the front foot at a stable target to create enough stride length.",
    ),
    "라인 스텝 드릴": (
        "Line Step Drill",
        "Step along a marked ground line to train weight transfer and stride direction.",
    ),
    "보폭 줄이기 컨트롤 드릴": (
        "Stride Shortening Control Drill",
        "Use a shorter-than-normal stride to control an overly long front-foot landing.",
    ),
    "균형 착지 드릴": (
        "Balanced Landing Drill",
        "Add a hold after front-foot landing to prevent the body from drifting forward.",
    ),
    "바닥 라인 밸런스 드릴": (
        "Ground Line Balance Drill",
        "Use a ground line to train head and shoulder alignment during the swing.",
    ),
    "미니밴드 힙 안정화 드릴": (
        "Mini-Band Hip Stability Drill",
        "Swing with a mini-band around the hips while keeping the core stable to reduce lateral sway.",
    ),
    "힙 텐션 릴리즈 드릴": (
        "Hip Tension Release Drill",
        "Use breathing and rhythm with a slight knee bend to reduce excessive side-to-side movement.",
    ),
    "스텝 백 풋워크 드릴": (
        "Step-Back Footwork Drill",
        "Progressively narrow stance and step length to keep the center of gravity from moving too far sideways.",
    ),
    "뒷다리 드라이브 드릴": (
        "Back-Leg Drive Drill",
        "Repeat movements that use the back leg and hips to stay loaded and maintain lower-body force.",
    ),
    "힙 힌지 리프트 드릴": (
        "Hip Hinge Lift Drill",
        "Use gradual lowering and rising through a hip hinge to create appropriate vertical movement at impact.",
    ),
    "코어 안정화 플랭크": (
        "Core Stability Plank",
        "Use plank and side-plank variations to improve core stability and reduce excessive center-of-gravity drop.",
    ),
    "미니 스쿼트 자세 유지 드릴": (
        "Mini Squat Hold Drill",
        "Maintain a controlled knee bend while swinging to prevent excessive vertical drop.",
    ),
    "시선 고정 드릴": (
        "Eye Fixation Drill",
        "Keep the eyes on the ball and impact point to reduce head movement.",
    ),
    "체어 로테이션 드릴": (
        "Chair Rotation Drill",
        "Practice isolated rotation while seated to reduce unnecessary upper-body and head movement.",
    ),
    "스쿼트 스윙 안정화 드릴": (
        "Squat Swing Stability Drill",
        "Swing from a shallow squat to reinforce head and upper-body alignment.",
    ),
    "바닥 마커 트래킹 드릴": (
        "Ground Marker Tracking Drill",
        "Track a marker on the ground during the swing to visually control head movement.",
    ),
    "로우 포지션 홀드 드릴": (
        "Low Position Hold Drill",
        "Pause before impact to keep front-knee flexion through stride landing.",
    ),
    "싱글 레그 스쿼트": (
        "Single-Leg Squat",
        "Build front-leg support with single-leg squats to increase knee flexion at impact.",
    ),
    "업라이트 포지션 리프트 드릴": (
        "Upright Position Lift Drill",
        "Slightly raise the torso and adjust stride length to avoid excessive front-knee bend.",
    ),
    "스트라이드 길이 컨트롤": (
        "Stride Length Control",
        "Gradually reduce stride distance to control excessive front-knee flexion at impact.",
    ),
    "몸통 기울임 드릴": (
        "Torso Tilt Drill",
        "Slowly repeat setup and load movements to create an appropriate torso tilt at load.",
    ),
    "힙 힌지 무브먼트": (
        "Hip Hinge Movement",
        "Practice hip hinge patterns to build an appropriate spine angle during load.",
    ),
    "업라이트 셋업 드릴": (
        "Upright Setup Drill",
        "Align the spine closer to vertical at stance to reduce excessive forward tilt.",
    ),
    "코어 앵글 드릴": (
        "Core Angle Drill",
        "Visually check core angle through setup and load to manage the upper limit of spine tilt.",
    ),
}

_METRIC_LABELS_EN = {
    "bat_speed": "bat speed",
    "attack_angle": "attack angle",
    "hip_shoulder_separation": "hip-shoulder separation",
    "hand_path_efficiency": "hand path efficiency",
    "stride_length_cm": "stride length",
    "cog_sway_cm": "center-of-gravity lateral sway",
    "cog_drop_cm": "center-of-gravity vertical drop",
    "head_stability_cm": "head stability",
    "front_knee_flexion_degrees": "front knee flexion",
    "spine_angle_degrees": "spine angle",
}


def localize_drill_recommendations(
    recommendations: list[DrillRecommendationResponse],
    locale: Locale,
) -> list[DrillRecommendationResponse]:
    """Return localized drill recommendation display fields."""
    if locale == "ko":
        return recommendations

    localized: list[DrillRecommendationResponse] = []
    for drill in recommendations:
        translated = _DRILL_TRANSLATIONS_EN.get(drill.drill_name)
        if translated:
            drill_name, description = translated
        else:
            drill_name, description = _generic_drill_text_en(drill)
        localized.append(
            DrillRecommendationResponse(
                drill_name=drill_name,
                target_metric=drill.target_metric,
                description=description,
                direction=drill.direction,
            )
        )
    return localized


def _generic_drill_text_en(drill: DrillRecommendationResponse) -> tuple[str, str]:
    metric = _METRIC_LABELS_EN.get(
        drill.target_metric,
        drill.target_metric.replace("_", " "),
    )
    if drill.direction == "above":
        description = (
            f"{metric.title()} is above the reference range. Work with a coach "
            "or video analysis process to design training that reduces the excessive movement."
        )
    elif drill.direction == "below":
        description = (
            f"{metric.title()} is below the reference range. Work with a coach "
            "or video analysis process to design training that builds the missing movement."
        )
    else:
        description = (
            f"No dedicated drill data is available for {metric}. Work with a coach "
            "to design a custom training plan."
        )
    return "Custom Training Plan Needed", description
