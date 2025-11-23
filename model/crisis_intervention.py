import re
from typing import Dict, Optional

# --- 1. 위기 단어 목록 ---
CRISIS_KEYWORDS = [
    "죽고싶", "죽을것같", "죽는게낫", "자살", "자해",
    "사라지고싶", "없어지고싶", "끝내고싶"
]
CRISIS_PATTERN = re.compile("|".join(CRISIS_KEYWORDS))

# --- 2. 간단한 감정 단어 치트 시트 ---
SIMPLE_EMOTION_MAP = {
    "두려워": "공포", "무서워": "공포", "겁나": "공포",
    "슬퍼": "슬픔", "우울해": "슬픔", "눈물 나": "슬픔",
    "화나": "분노", "열받아": "분노", "짜증나": "분노",
    "기뻐": "행복", "행복해": "행복", "신나": "행복",
}

# ================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ '반어법 탐지기'가 추가되었습니다! ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# --- 3. 부정적 맥락의 긍정 단어 패턴 ---
CONTRAST_PATTERNS = {
    # '슬픔'으로 판단해야 하는 경우
    "슬픔": {
        "positive_keywords": ["행복", "즐거워", "웃", "좋아보여"],
        "negative_triggers": ["나빼고", "나만 빼고", "너는", "쟤는", "다들"]
    }
}
# ================================================================


def check_crisis_message(text: str) -> Optional[Dict]:
    """텍스트에 위기 신호가 있는지 확인합니다."""
    if CRISIS_PATTERN.search(text.replace(" ", "")):
        print(f"[CRISIS DETECTED] 위기 신호 감지: '{text}'")
        return {"emotions": {'공포': 0.0, '놀람': 0.1, '분노': 0.8, '슬픔': 0.9, '중립': 0.0, '행복': 0.0, '혐오': 0.2}}
    return None

def check_simple_emotion(text: str) -> Optional[Dict]:
    """텍스트에 명백한 감정 단어가 있는지 치트 시트에서 확인합니다."""
    for keyword, emotion in SIMPLE_EMOTION_MAP.items():
        if keyword in text:
            print(f"[SIMPLE EMOTION DETECTED] '{keyword}' -> '{emotion}'으로 판단")
            emotions = {e: 0.0 for e in ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']}
            emotions[emotion] = 0.9
            return {"emotions": emotions}
    return None

# ================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ '반어법 탐지기'를 실행하는 새 함수 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
def check_contrast_emotion(text: str) -> Optional[Dict]:
    """긍정 단어가 부정적 맥락에서 쓰였는지 확인합니다."""
    for target_emotion, patterns in CONTRAST_PATTERNS.items():
        has_positive = any(pk in text for pk in patterns["positive_keywords"])
        has_negative = any(nt in text for nt in patterns["negative_triggers"])
        
        if has_positive and has_negative:
            print(f"[CONTRAST DETECTED] '{text}' -> '{target_emotion}'으로 판단")
            emotions = {e: 0.0 for e in ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']}
            emotions[target_emotion] = 0.9 # 목표 감정을 높게 설정
            if target_emotion == "슬픔": emotions["분노"] = 0.3 # 슬픔과 분노가 함께 나타나는 경우가 많음
            return {"emotions": emotions}
    return None
# ================================================================

