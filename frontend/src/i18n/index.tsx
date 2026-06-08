import {
  createContext,
  useCallback,
  useContext,
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
      "정확한 분석을 위해 프로필 정보를 입력하세요. 건너뛰기를 누르면 기본값으로 분석합니다.",
    skipAndStart: "건너뛰고 분석 시작",
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
    drillsAria: "드릴 추천",
    drillsTitle: "추천 드릴",
    noDrills: "추천 드릴이 없습니다.",
  },
  overlay: {
    title: "오버레이 비디오",
    aria: "오버레이 비디오 플레이어",
    unsupported: "이 브라우저는 비디오 재생을 지원하지 않습니다.",
    unavailable: "오버레이 비디오를 사용할 수 없습니다.",
  },
  metrics: {
    title: "메트릭",
    aria: "메트릭 테이블",
    empty: "표시할 메트릭이 없습니다.",
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
      "Enter profile details for a more accurate analysis. If you skip this step, default values will be used.",
    skipAndStart: "Skip and Start Analysis",
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
    drillsAria: "Drill recommendations",
    drillsTitle: "Recommended Drills",
    noDrills: "No drill recommendations.",
  },
  overlay: {
    title: "Overlay Video",
    aria: "Overlay video player",
    unsupported: "This browser does not support video playback.",
    unavailable: "Overlay video is unavailable.",
  },
  metrics: {
    title: "Metrics",
    aria: "Metrics table",
    empty: "No metrics to display.",
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
