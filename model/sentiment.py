# sentiment.py (경량 배포용: 규칙 기반 + 위기 탐지)
import os
import re
import numpy as np
from typing import Dict

from model.crisis_intervention import (
    check_crisis_message, check_simple_emotion, check_contrast_emotion
)

EMO_DIR = "models/emotions_kor"
EMOS = ["공포", "놀람", "분노", "슬픔", "중립", "행복", "혐오"]

# ==== 규칙용 사전들 (기존 것 그대로) ====
RULE_BOOST = {
    "공포": 0.18,
    "놀람": 0.15,
    "분노": 0.25,
    "슬픔": 0.30,
    "중립": 0.10,
    "행복": 0.28,
    "혐오": 0.22,
}
INTENSIFIERS = ["너무", "매우", "엄청", "정말", "진짜", "완전", "되게", "개", "존나", "졸라", "대박", "킹받", "개빡", "개무섭"]
DIMINISHERS = ["좀", "조금", "약간", "그럭저럭", "그냥", "살짝", "괜찮"]
NEGATIONS = ["아니", "안 ", "않", "별로", "전혀", "그렇진", "그렇지 않", "아냐", "아닌", "없", "못 "]
CONTRASTORS = ["하지만", "그래도", "근데", "그러나"]

LEX = {
    "슬픔": [
        "허망",
        "허무",
        "허탈",
        "현타",
        "상실",
        "좌절",
        "패배감",
        "눈물",
        "울컥",
        "오열",
        "슬프",
        "비통",
        "우울",
        "공허",
        "외롭",
        "적막",
        "막막",
        "무력감",
        "힘들",
        "지쳤",
        "서글프",
    ],
    "공포": [
        "불안",
        "걱정",
        "두렵",
        "겁나",
        "겁이",
        "초조",
        "긴장",
        "덜컥",
        "불길",
        "무섭",
        "소름",
        "불편하",
        "초조하",
    ],
    "분노": [
        "분노",
        "화가",
        "화남",
        "짜증",
        "억울",
        "부들",
        "열받",
        "빡치",
        "빡쳐",
        "빡침",
        "성질",
        "화딱지",
        "개빡",
        "미치겠",
        "X같",
        "좆같",
        "씨발",
        "시발",
    ],
    "혐오": ["혐오", "역겹", "징그", "구역질", "불결", "더럽", "토나", "비위가"],
    "놀람": ["놀랐", "충격", "경악", "어이없", "헐", "와...", "세상에", "헉"],
    "행복": ["행복", "기쁨", "기뻐", "좋아", "좋네", "행복하", "감사", "다행", "설레", "설렘", "신나", "벅차"],
    "중립": ["그냥", "그럭저럭", "보통", "평범", "무난", "그렇구나", "음...", "음.."],
}

LONELY_PATTERNS = [
    r"외로(워|움|운|울)",
    r"혼자(라|서)",
    r"친구가 없",
    r"연락이 없",
    r"나밖에 없",
    r"공허(해|함|하다)",
]
FEAR_HINTS = ["불안", "무섭", "두렵", "초조", "긴장"]

# ==== 유틸 함수들 ====
def _contains_any(text: str, words) -> bool:
    return any(w in text for w in words)


def _negated_near(text: str, keyword: str, window: int = 8) -> bool:
    """키워드 주변에 부정 표현이 있는지 확인"""
    idx = text.find(keyword)
    if idx == -1:
        return False
    start = max(0, idx - window)
    snippet = text[start : idx + len(keyword) + window]
    return any(ng in snippet for ng in NEGATIONS)


def adjust_with_rules(text: str, probs: Dict[str, float]) -> Dict[str, float]:
    """
    base probs(모두 0일 수도 있음)를 받아서
    키워드/맥락 규칙으로 감정 분포를 보정한다.
    """
    out = {k: float(max(0.0, probs.get(k, 0.0))) for k in EMOS}
    t = (text or "").strip()
    if not t:
        s = sum(out.values()) or 1.0
        return {k: v / s for k, v in out.items()}

    # 1) 키워드 가산 (+부정 약화)
    for emo, kws in LEX.items():
        for kw in kws:
            if kw in t:
                boost = RULE_BOOST.get(emo, 0.2)
                if _negated_near(t, kw):
                    boost *= 0.35
                out[emo] = min(1.0, out.get(emo, 0.0) + boost)

    # 2) 외로움/고독 패턴 → 슬픔 크게 보강
    if any(re.search(p, t) for p in LONELY_PATTERNS):
        out["슬픔"] = min(1.0, out.get("슬픔", 0.0) + 0.25)
        # 공포 단서 거의 없으면 슬픔 추가 + 공포 약화
        if not any(h in t for h in FEAR_HINTS):
            out["슬픔"] = min(1.0, out.get("슬픔", 0.0) + 0.15)
            out["공포"] = max(0.0, out.get("공포", 0.0) - 0.10)

    # 3) 허무/상실 + 불안 공존 시 슬픔 쪽으로 가중
    if _contains_any(t, ["허망", "허무", "허탈", "상실", "공허", "현타"]) and _contains_any(
        t, ["불안", "걱정", "두렵", "무섭"]
    ):
        out["슬픔"] = min(1.0, out.get("슬픔", 0.0) + 0.20)

    # 4) 강도 보정 (너무/조금 등)
    s = t
    if any(i in s for i in INTENSIFIERS):
        for k in EMOS:
            out[k] = min(1.0, out.get(k, 0.0) * 1.2)
    if any(d in s for d in DIMINISHERS):
        for k in EMOS:
            out[k] = max(0.0, out.get(k, 0.0) * 0.8)

    # 5) 전체 정규화 + 극단치 클램프
    vec = np.array([max(1e-8, out[k]) for k in EMOS], dtype=float)
    s = float(vec.sum())
    if s > 0:
        vec /= s
    vec = np.clip(vec, 0.0, 0.95)
    vec = vec / (vec.sum() or 1.0)

    return {EMOS[i]: float(round(vec[i], 6)) for i in range(len(EMOS))}


# ---- 배포용: 모델 없이 동작하는 init/analyze ----
def init_models():
    """배포용 경량 모드: 별도 학습 모델 없이 규칙 기반만 사용."""
    print(
        "[sentiment] (light) init_models(): no transformer model loaded – using rule-based analyzer only."
    )


def analyze_emotion(text: str) -> Dict:
    """
    1) 위기 문장
    2) 반어법/대조 맥락
    3) 단순 감정 단어
    를 먼저 확인하고, 남은 경우 규칙 기반 감정 분포를 계산.
    """
    text = (text or "").strip()
    if not text:
        return {"emotions": {e: 0.0 for e in EMOS}}

    # 1) 위기 메시지 우선
    crisis_result = check_crisis_message(text)
    if crisis_result:
        return crisis_result

    # 2) 반어법/대조 맥락
    contrast_result = check_contrast_emotion(text)
    if contrast_result:
        return contrast_result

    # 3) "슬퍼", "화나" 같은 직접 표현
    simple_result = check_simple_emotion(text)
    if simple_result:
        return simple_result

    # 4) 규칙 기반으로 감정 분포 계산
    base = {e: 0.0 for e in EMOS}
    emotions = adjust_with_rules(text, base)
    return {"emotions": emotions}
