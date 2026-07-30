import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

export type Language = "ko" | "en";

type TranslationValue = string | TranslationTree;
type TranslationTree = { [key: string]: TranslationValue };
type TranslationParams = Record<string, string | number>;

const STORAGE_KEY = "myswing.language";

const ko = {
  app: {
    language: "언어",
    korean: "한국어",
    english: "English",
  },
  uploadPage: {
    title: "MySwing - AI 야구 스윙 분석",
    intro:
      "한 번의 스윙만 담긴 짧은 야구 영상을 업로드하세요. 권장 길이는 3~7초이며, 약 5초 영상이 가장 적합합니다.",
    profileTitle: "사용자 프로필",
    profileIntro:
      "분석을 시작하려면 프로필 정보를 입력하세요.",
    uploadedVideo: "업로드한 영상",
    starting: "분석을 시작하는 중...",
    startError: "분석 시작에 실패했습니다.",
  },
  analysisPage: {
    title: "분석 리포트",
    missingId: "유효한 분석 ID가 제공되지 않았습니다.",
    newAnalysis: "새 영상 분석하기",
    loading: "리포트를 불러오는 중...",
    retry: "다시 시도하기",
    reportError: "리포트 조회에 실패했습니다.",
    analysisError: "분석에 실패했습니다.",
  },
  uploader: {
    preparing: "업로드 URL 요청 중...",
    uploading: "업로드 중...",
    fetchingMetadata: "메타데이터 분석 중...",
    complete: "업로드 완료",
    error: "업로드 실패",
    unsupportedType:
      "지원하지 않는 형식입니다: {type}. MP4, MOV, AVI만 지원합니다.",
    fileTooLarge: "파일 크기가 너무 큽니다. 최대 {limitMb}MB",
    dropzoneLabel: "비디오 파일을 드롭하거나 클릭하여 선택",
    hint: "한 번의 스윙만 담긴 짧은 영상을 업로드하세요. 권장 길이는 3~7초이며, 약 5초 영상이 가장 적합합니다.",
    formats: "MP4, MOV, AVI - 최대 500MB, 최대 10초. 긴 영상은 한 번의 스윙만 남기고 잘라주세요.",
    inputLabel: "한 번의 스윙만 담긴 비디오 파일 선택",
    retry: "다시 시도",
    uploadAnother: "다른 파일 업로드",
    durationTooLong: "영상 길이가 10초를 초과했습니다. 한 번의 스윙만 담기도록 클립을 잘라 업로드하세요.",
    durationWarning: "권장 길이는 3~7초입니다. 약 5초 분량의 한 번의 스윙 클립이 가장 적합합니다.",
    durationCheckUnavailable: "브라우저에서 영상 길이를 확인할 수 없습니다. 서버에서 다시 검증합니다.",
  },
  profile: {
    loading: "프로필 불러오는 중...",
    title: "사용자 프로필",
    hint: "분석 정확도 향상을 위해 신체 정보와 타격 특성을 입력해주세요.",
    requiredHint: " * 표시는 필수 항목",
    requiredSection: "필수 정보",
    optionalSection: "선택 정보",
    height: "키 (height)",
    batLength: "배트 길이 (bat length)",
    battingDirection: "타격 방향 (batting direction)",
    weight: "체중 (weight)",
    cameraDirection: "촬영 방향 (camera direction)",
    ageGroup: "연령대 (age group)",
    level: "수준 (level)",
    batWeight: "배트 무게 (bat weight)",
    validRange: "유효 범위: {min}-{max}{unit}",
    batLengthRange: "유효 범위: {minIn}-{maxIn}인치 또는 {minCm}-{maxCm}cm",
    recommended: "권장: 분석 정확도를 위해 입력해주세요.",
    selectRequired: "선택해주세요",
    selectOptional: "선택 안 함",
    agePlaceholder: "예: 20대, U-19",
    inchOrCm: "in 또는 cm",
    submitError: "저장 실패: {message}",
    saved: "프로필이 저장되었습니다.",
    saving: "저장 중...",
    save: "프로필 저장",
    options: {
      battingLeft: "왼손 타자 (Left)",
      battingRight: "오른손 타자 (Right)",
      cameraFront: "정면 (Front)",
      cameraSide: "측면 (Side)",
      cameraRear: "후면 (Rear)",
      professional: "프로 (Professional)",
      college: "대학 (College)",
      highSchool: "고교 (High school)",
      recreational: "동호인 (Recreational)",
    },
    errors: {
      heightRequired: "키(height)는 필수 입력 항목입니다.",
      heightNumber: "키는 숫자로 입력해주세요.",
      heightRange: "키는 {min}-{max}cm 사이여야 합니다.",
      batLengthRequired: "배트 길이(bat length)는 필수 입력 항목입니다.",
      batLengthNumber: "배트 길이는 숫자로 입력해주세요.",
      batLengthRange:
        "배트 길이는 {minIn}-{maxIn}인치 또는 {minCm}-{maxCm}cm 사이여야 합니다.",
      battingDirectionRequired:
        "타격 방향(batting direction)은 필수 입력 항목입니다.",
      battingDirectionInvalid: "타격 방향은 left 또는 right 여야 합니다.",
      weightPositive: "체중은 0보다 큰 숫자여야 합니다.",
      batWeightNumber: "배트 무게는 숫자로 입력해주세요.",
      batWeightRange: "배트 무게는 {min}-{max}oz 사이여야 합니다.",
    },
  },
  status: {
    aria: "분석 진행 상태",
    title: "분석 진행 상태",
    currentPhase: "현재 단계: {phase}",
    progress: "분석 진행률",
    phases: {
      pending: "대기 중",
      preprocessing: "비디오 전처리 중",
      analyzing: "스윙 분석 중",
      evaluating: "메트릭 평가 중",
      generating_report: "리포트 생성 중",
      completed: "분석 완료",
      failed: "분석 실패",
    },
  },
  report: {
    aria: "분석 결과 리포트",
    title: "분석 리포트",
    analysisId: "분석 ID: {id}",
    createdAt: "생성일 {date}",
    multipleSwingsAria: "다중 스윙 감지 안내",
    multipleSwings:
      "영상에서 여러 번의 스윙 동작이 감지되어, 가장 먼저 시작된 유의미한 스윙만 분석했습니다.",
    multipleSwingsWithCount:
      "영상에서 스윙 동작 후보 {count}개가 감지되어, 가장 먼저 시작된 유의미한 스윙만 분석했습니다.",
    drillsAria: "드릴 추천",
    drillsTitle: "추천 드릴",
    noDrills: "추천 드릴이 없습니다.",
  },
  metricLabels: {
    batSpeed: "배트 속도",
    attackAngle: "어택 앵글",
    handPathEfficiency: "핸드 패스 효율",
    kinematicChain: "키네마틱 체인",
    hipShoulderSeparation: "힙-숄더 분리",
  },
  overlay: {
    title: "오버레이 비디오",
    aria: "오버레이 비디오 플레이어",
    unsupported: "이 브라우저는 비디오 재생을 지원하지 않습니다.",
    unavailable: "오버레이 비디오를 사용할 수 없습니다.",
    phases: "구간",
    jumpToPhase: "{phase} 구간으로 이동",
  },
  metrics: {
    title: "검증된 기준 평가",
    aria: "검증된 기준 평가 테이블",
    empty: "현재 사용할 수 있는 검증된 프로 기준 평가가 없습니다.",
    metric: "메트릭",
    measured: "측정값",
    referenceRange: "참조 범위",
    deviation: "편차",
    rating: "평가",
    ratings: {
      below_range: "기준 미만",
      within_range: "적정",
      above_range: "기준 초과",
    },
  },
  biomechanics: {
    title: "투영 생체역학 진단값",
    aria: "투영 생체역학 진단값",
    disclaimer:
      "단일 카메라 영상에서 계산한 2D 투영 진단값입니다. 검증된 프로 기준 등급이나 실제 3D 측정값이 아닙니다.",
    empty: "이 영상에서 보고할 수 있는 투영 생체역학 값이 없습니다.",
    unmeasurableTitle: "측정하지 못한 항목",
    phaseSource: "스윙 구간 판정 근거",
    reportingResolution: "보고 해상도: {value}{unit}",
    metrics: {
      batSpeed: "배트 속도 (2D 투영)",
      attackAngle: "어택 앵글 (부호 있는 2D 투영)",
      handPathEfficiency: "핸드 패스 직선성 (프록시)",
      strideLength: "발목/스탠스 폭 (기존 스트라이드 프록시)",
      cogSway: "골반 좌우 이동 (기존 무게중심 프록시)",
      cogDrop: "골반 수직 이동 (기존 무게중심 프록시)",
      headStability: "머리 중심 이동량 (2D 투영)",
      frontKneeExtension: "앞 무릎 신전 각도 (2D 투영)",
      frontKneeFlexion: "앞 무릎 굴곡 각도 (2D 투영)",
      spineAngle: "어깨-골반 몸통 기울기 (기존 척추 프록시)",
    },
    phaseSources: {
      poseClassifier: "포즈 기반 분류기",
      partialPoseClassifier: "부분 포즈 기반 분류기",
      mixedPoseAndObservedBatContact:
        "포즈 기반 구간 + 검출기가 관측한 배트 접촉",
      mixedPoseClassifierAndPoseMotionContact:
        "포즈 기반 구간 + 포즈 동작 기반 접촉 추정",
      poseMotionContactOnly: "포즈 동작 기반 접촉 추정만 사용",
      observedBatContactOnly: "검출기가 관측한 배트 접촉만 사용",
      unavailable: "판정 불가",
      notRecorded: "기록되지 않음 (이전 분석)",
    },
  },
  drill: {
    aria: "드릴 추천: {name}",
    target: "대상: {metric}",
    direction: "방향: {direction}",
    directions: {
      below: "기준 미달",
      above: "기준 초과",
      generic: "맞춤 안내",
    },
  },
  comparison: {
    title: "사용자 vs 프로 비교 (단계 지속 시간)",
    aria: "사용자 vs 프로 비교 뷰",
    user: "사용자",
    reference: "프로 참조",
    empty: "비교할 스윙 단계 데이터가 없습니다.",
    phase: "단계",
    userMs: "사용자 (ms)",
    referenceMs: "프로 (ms)",
    compare: "비교",
    userDuration: "사용자 {phase} 지속시간",
    referenceDuration: "프로 {phase} 지속시간",
  },
  phases: {
    stance: "준비",
    load: "로드",
    stride: "스트라이드",
    rotation: "회전",
    impact: "임팩트",
    follow_through: "팔로스루",
  },
  improvements: {
    title: "개선이 필요한 영역 (상위 3개)",
    aria: "개선이 필요한 영역",
    empty: "개선이 필요한 영역이 식별되지 않았습니다.",
    rank: "순위 {rank}",
    currentTarget: "현재값 {current} / 목표 {min} - {max}",
    deviation: "편차 {value}%",
  },
  trend: {
    title: "메트릭 추이",
    aria: "메트릭 추이",
    insufficient:
      "트렌드 분석을 위해서는 최소 {min}회 이상의 분석 기록이 필요합니다 (현재 {total}회).",
    empty: "트렌드 데이터가 없습니다.",
    chartAria: "{metric} 추이 차트",
    metric: "메트릭",
    allMetrics: "전체",
    dateRange: "{start} - {end}",
    latestChange: "최근 변화 {value}",
  },
  quality: {
    aria: "비디오 품질 검증 결과",
    title: "품질 검증 결과",
    brightness: "밝기",
    framing: "스윙 아크 가시성",
    resolution: "해상도",
    fps: "프레임레이트 안정성",
    sufficient: "충분",
    low: "낮음",
    variation: "편차 {value}%",
    pass: "통과",
    warning: "경고",
  },
  metadata: {
    aria: "비디오 정보",
    thumbnailAlt: "{fileName} 썸네일",
    noThumbnail: "썸네일 사용 불가",
    fileName: "파일명",
    duration: "길이",
    resolution: "해상도",
    size: "크기",
  },
} satisfies TranslationTree;

const en = {
  app: {
    language: "Language",
    korean: "한국어",
    english: "English",
  },
  uploadPage: {
    title: "MySwing - AI Baseball Swing Analysis",
    intro:
      "Upload a short baseball video containing exactly one swing. Recommended length is 3–7 seconds; around 5 seconds is ideal.",
    profileTitle: "User Profile",
    profileIntro:
      "Enter your profile details to start the analysis.",
    uploadedVideo: "Uploaded Video",
    starting: "Starting analysis...",
    startError: "Failed to start analysis.",
  },
  analysisPage: {
    title: "Analysis Report",
    missingId: "A valid analysis ID was not provided.",
    newAnalysis: "Analyze New Video",
    loading: "Loading report...",
    retry: "Try Again",
    reportError: "Failed to load the report.",
    analysisError: "Analysis failed.",
  },
  uploader: {
    preparing: "Requesting upload URL...",
    uploading: "Uploading...",
    fetchingMetadata: "Analyzing metadata...",
    complete: "Upload complete",
    error: "Upload failed",
    unsupportedType:
      "Unsupported file type: {type}. Only MP4, MOV, and AVI are supported.",
    fileTooLarge: "File is too large. Maximum {limitMb}MB",
    dropzoneLabel: "Drop a video file or click to select",
    hint: "Upload a short video containing exactly one swing. Recommended length is 3–7 seconds; around 5 seconds is ideal.",
    formats: "MP4, MOV, AVI - up to 500MB and 10 seconds. Trim long videos to a single swing.",
    inputLabel: "Select a single-swing video file",
    retry: "Try Again",
    uploadAnother: "Upload Another File",
    durationTooLong: "This video is longer than 10 seconds. Trim it to a short clip containing exactly one swing.",
    durationWarning: "Recommended length is 3–7 seconds; around 5 seconds with exactly one swing is ideal.",
    durationCheckUnavailable: "The browser could not read the video duration. The server will validate it again.",
  },
  profile: {
    loading: "Loading profile...",
    title: "User Profile",
    hint: "Enter body information and batting details to improve analysis accuracy.",
    requiredHint: " * marks required fields",
    requiredSection: "Required Information",
    optionalSection: "Optional Information",
    height: "Height",
    batLength: "Bat Length",
    battingDirection: "Batting Direction",
    weight: "Weight",
    cameraDirection: "Camera Direction",
    ageGroup: "Age Group",
    level: "Level",
    batWeight: "Bat Weight",
    validRange: "Valid range: {min}-{max}{unit}",
    batLengthRange: "Valid range: {minIn}-{maxIn} in or {minCm}-{maxCm} cm",
    recommended: "Recommended for better analysis accuracy.",
    selectRequired: "Select an option",
    selectOptional: "None selected",
    agePlaceholder: "e.g. 20s, U-19",
    inchOrCm: "in or cm",
    submitError: "Save failed: {message}",
    saved: "Profile saved.",
    saving: "Saving...",
    save: "Save Profile",
    options: {
      battingLeft: "Left-handed hitter",
      battingRight: "Right-handed hitter",
      cameraFront: "Front",
      cameraSide: "Side",
      cameraRear: "Rear",
      professional: "Professional",
      college: "College",
      highSchool: "High school",
      recreational: "Recreational",
    },
    errors: {
      heightRequired: "Height is required.",
      heightNumber: "Height must be a number.",
      heightRange: "Height must be between {min}-{max}cm.",
      batLengthRequired: "Bat length is required.",
      batLengthNumber: "Bat length must be a number.",
      batLengthRange:
        "Bat length must be between {minIn}-{maxIn} inches or {minCm}-{maxCm}cm.",
      battingDirectionRequired: "Batting direction is required.",
      battingDirectionInvalid: "Batting direction must be left or right.",
      weightPositive: "Weight must be a number greater than 0.",
      batWeightNumber: "Bat weight must be a number.",
      batWeightRange: "Bat weight must be between {min}-{max}oz.",
    },
  },
  status: {
    aria: "Analysis progress",
    title: "Analysis Progress",
    currentPhase: "Current phase: {phase}",
    progress: "Analysis progress",
    phases: {
      pending: "Pending",
      preprocessing: "Preprocessing video",
      analyzing: "Analyzing swing",
      evaluating: "Evaluating metrics",
      generating_report: "Generating report",
      completed: "Analysis complete",
      failed: "Analysis failed",
    },
  },
  report: {
    aria: "Analysis report",
    title: "Analysis Report",
    analysisId: "Analysis ID: {id}",
    createdAt: "Created {date}",
    multipleSwingsAria: "Multiple swing detection notice",
    multipleSwings:
      "Multiple swing motions were detected in this video; only the earliest substantial swing was analyzed.",
    multipleSwingsWithCount:
      "{count} swing-motion candidates were detected in this video; only the earliest substantial swing was analyzed.",
    drillsAria: "Drill recommendations",
    drillsTitle: "Recommended Drills",
    noDrills: "No drill recommendations.",
  },
  metricLabels: {
    batSpeed: "Bat Speed",
    attackAngle: "Attack Angle",
    handPathEfficiency: "Hand Path Efficiency",
    kinematicChain: "Kinematic Chain",
    hipShoulderSeparation: "Hip-Shoulder Separation",
  },
  overlay: {
    title: "Overlay Video",
    aria: "Overlay video player",
    unsupported: "This browser does not support video playback.",
    unavailable: "Overlay video is unavailable.",
    phases: "Phases",
    jumpToPhase: "Jump to {phase}",
  },
  metrics: {
    title: "Validated Reference Evaluations",
    aria: "Validated reference evaluations table",
    empty: "No validated professional-reference evaluations are currently available.",
    metric: "Metric",
    measured: "Measured Value",
    referenceRange: "Reference Range",
    deviation: "Deviation",
    rating: "Rating",
    ratings: {
      below_range: "Below Range",
      within_range: "Within Range",
      above_range: "Above Range",
    },
  },
  biomechanics: {
    title: "Projected Biomechanics Diagnostics",
    aria: "Projected biomechanics diagnostics",
    disclaimer:
      "These are monocular 2D projected diagnostics, not validated professional-reference grades or true 3D measurements.",
    empty: "No projected biomechanics values could be reported from this video.",
    unmeasurableTitle: "Could Not Measure",
    phaseSource: "Swing-phase source",
    reportingResolution: "Reporting resolution: {value}{unit}",
    metrics: {
      batSpeed: "Bat speed (2D projected)",
      attackAngle: "Attack angle (signed 2D projection)",
      handPathEfficiency: "Hand-path straightness (proxy)",
      strideLength: "Ankle/stance span (legacy stride proxy)",
      cogSway: "Pelvis horizontal translation (legacy CoG proxy)",
      cogDrop: "Pelvis vertical translation (legacy CoG proxy)",
      headStability: "Head-center translation (2D projected)",
      frontKneeExtension: "Front-knee extension angle (2D projected)",
      frontKneeFlexion: "Front-knee flexion angle (2D projected)",
      spineAngle: "Shoulder-pelvis trunk lean (legacy spine proxy)",
    },
    phaseSources: {
      poseClassifier: "Pose-derived classifier",
      partialPoseClassifier: "Partial pose-derived classifier",
      mixedPoseAndObservedBatContact:
        "Pose-derived phases + detector-observed bat contact",
      mixedPoseClassifierAndPoseMotionContact:
        "Pose-derived phases + pose-motion contact estimate",
      poseMotionContactOnly: "Pose-motion contact estimate only",
      observedBatContactOnly: "Detector-observed bat contact only",
      unavailable: "Unavailable",
      notRecorded: "Not recorded (legacy analysis)",
    },
  },
  drill: {
    aria: "Drill recommendation: {name}",
    target: "Target: {metric}",
    direction: "Direction: {direction}",
    directions: {
      below: "Below Range",
      above: "Above Range",
      generic: "Custom Guidance",
    },
  },
  comparison: {
    title: "User vs Pro Comparison (Phase Duration)",
    aria: "User vs pro comparison view",
    user: "User",
    reference: "Pro Reference",
    empty: "No swing phase data to compare.",
    phase: "Phase",
    userMs: "User (ms)",
    referenceMs: "Pro (ms)",
    compare: "Comparison",
    userDuration: "User {phase} duration",
    referenceDuration: "Pro {phase} duration",
  },
  phases: {
    stance: "Stance",
    load: "Load",
    stride: "Stride",
    rotation: "Rotation",
    impact: "Impact",
    follow_through: "Follow Through",
  },
  improvements: {
    title: "Top Improvement Areas",
    aria: "Improvement areas",
    empty: "No improvement areas were identified.",
    rank: "Rank {rank}",
    currentTarget: "Current {current} / Target {min} - {max}",
    deviation: "Deviation {value}%",
  },
  trend: {
    title: "Metric Trends",
    aria: "Metric trends",
    insufficient:
      "At least {min} analysis records are required for trend analysis (currently {total}).",
    empty: "No trend data.",
    chartAria: "{metric} trend chart",
    metric: "Metric",
    allMetrics: "All",
    dateRange: "{start} - {end}",
    latestChange: "Latest change {value}",
  },
  quality: {
    aria: "Video quality check result",
    title: "Quality Check Result",
    brightness: "Brightness",
    framing: "Swing Arc Visibility",
    resolution: "Resolution",
    fps: "Frame Rate Stability",
    sufficient: "Sufficient",
    low: "Low",
    variation: "Variation {value}%",
    pass: "Pass",
    warning: "Warning",
  },
  metadata: {
    aria: "Video information",
    thumbnailAlt: "{fileName} thumbnail",
    noThumbnail: "Thumbnail unavailable",
    fileName: "File Name",
    duration: "Duration",
    resolution: "Resolution",
    size: "Size",
  },
} satisfies TranslationTree;

const translations: Record<Language, TranslationTree> = { ko, en };

interface I18nContextValue {
  language: Language;
  setLanguage: (language: Language) => void;
  t: (key: string, params?: TranslationParams) => string;
}

const I18nContext = createContext<I18nContextValue | null>(null);

const fallbackI18n: I18nContextValue = {
  language: "ko",
  setLanguage: () => undefined,
  t: (key, params) => {
    const value = lookup(translations.ko, key) ?? key;
    return interpolate(value, params);
  },
};

function initialLanguage(): Language {
  if (typeof window === "undefined") return "ko";
  const stored = window.localStorage.getItem(STORAGE_KEY);
  return stored === "en" || stored === "ko" ? stored : "ko";
}

function lookup(tree: TranslationTree, key: string): string | null {
  let current: TranslationValue | undefined = tree;
  for (const part of key.split(".")) {
    if (typeof current !== "object" || current === null) return null;
    current = current[part];
  }
  return typeof current === "string" ? current : null;
}

function interpolate(value: string, params?: TranslationParams): string {
  if (!params) return value;
  return value.replace(/\{(\w+)\}/g, (_, key: string) =>
    params[key] == null ? `{${key}}` : String(params[key]),
  );
}

export function I18nProvider({ children }: { children: ReactNode }) {
  const [language, setLanguageState] = useState<Language>(initialLanguage);

  useEffect(() => {
    if (typeof document !== "undefined") {
      document.documentElement.lang = language;
    }
  }, [language]);

  const setLanguage = useCallback((next: Language) => {
    setLanguageState(next);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(STORAGE_KEY, next);
    }
  }, []);

  const t = useCallback(
    (key: string, params?: TranslationParams) => {
      const value =
        lookup(translations[language], key) ?? lookup(translations.ko, key) ?? key;
      return interpolate(value, params);
    },
    [language],
  );

  const value = useMemo(
    () => ({ language, setLanguage, t }),
    [language, setLanguage, t],
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useTranslation(): I18nContextValue {
  return useContext(I18nContext) ?? fallbackI18n;
}
