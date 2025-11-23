# sentiment.py (미세 튜닝＋외로움 보강)
import os
import re
import numpy as np
import torch
from typing import Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from model.crisis_intervention import (
    check_crisis_message, check_simple_emotion, check_contrast_emotion
)

EMO_DIR = 'models/emotions_kor'
EMOS = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']

_e_tok = _e_mod = None

# ---- 샤프닝/규칙 가산 기본값
GAMMA = 1.25
RULE_BOOST = {
    '공포': 0.18, '놀람': 0.15, '분노': 0.25,
    '슬픔': 0.30, '중립': 0.10, '행복': 0.28, '혐오': 0.22,
}
INTENSIFIERS = ['너무','매우','엄청','정말','진짜','완전','되게','개','존나','졸라','대박','킹받','개빡','개무섭']
DIMINISHERS = ['좀','조금','약간','그럭저럭','그냥','살짝','괜찮']
NEGATIONS = ['아니','안 ','않','별로','전혀','그렇진','그렇지 않','아냐','아닌','없','못 ']
CONTRASTORS = ['하지만','그래도','근데','그러나']

LEX = {
    '슬픔': ['허망','허무','허탈','현타','상실','좌절','패배감','눈물','울컥','오열','슬프','비통','우울','공허','외롭','적막','막막','무력감','힘들','지쳤','서글프'],
    '공포': ['불안','걱정','두렵','겁나','겁이','초조','긴장','덜컥','불길','무섭','소름','불편하','초조하'],
    '분노': ['분노','화가','화남','짜증','억울','부들','열받','빡치','빡쳐','빡침','성질','화딱지','개빡','미치겠','X같','좆같','씨발','시발'],
    '혐오': ['혐오','역겹','징그','구역질','불결','더럽','토나','비위가'],
    '놀람': ['놀랐','충격','경악','어이없','헐','세상에','말도안돼','믿기지','깜짝','헉'],
    '행복': ['행복','기쁘','좋아','좋다','설렘','설렌','즐겁','뿌듯','감사','사랑','사랑해','편안','든든','위로가 돼','위안'],
}

EMOJI_MAP = {
    '슬픔': ['😭','😢','ㅜㅜ','ㅠㅠ','흑흑','엉엉','T_T','TT',';_;'],
    '공포': ['😱','😨','무서워','덜덜','ㄷㄷ','무섭'],
    '분노': ['😡','🤬','화나','빡침','열받'],
    '행복': ['😊','😄','😁','😍','🤗','ㅎㅎ','ㅋㅋ','^^','^_^'],
    '놀람': ['😲','😮','헉','헐','와우'],
    '혐오': ['🤢','🤮'],
}

FEAR_QUESTIONS = ['어떡하','어쩌','불안해','괜찮을까','무서울','죽겠']
SADNESS_PHRASES = ['왜 나만','포기하고 싶','살기 싫','더는 못','희망이 없','희망 없다']

# ▶ 외로움/고독 패턴(정규식)
LONELY_PATTERNS = [
    r'외롭[다요]?', r'외로움', r'쓸쓸', r'고독',
    r'혼자(야|라서|만|서)',                    # 혼자야/혼자라서/혼자만/혼자서
    r'(친한|가까운)[^가-힣]{0,2}사람이 없',      # 친한 사람이 없어
    r'연락(할|하[는ㄴ]) 사람이 없',             # 연락할/하는 사람이 없
]
FEAR_HINTS = ['불안','걱정','두렵','겁나','무섭','초조','긴장']

def _contains_any(text: str, words: list[str]) -> bool:
    t = text.lower()
    return any(w.lower() in t for w in words)

def _negated_near(text: str, keyword: str, window: int = 3) -> bool:
    t = text
    idx = t.find(keyword)
    if idx == -1:
        return False
    left = t[max(0, idx - window*2): idx+1]
    return any(ng in left for ng in NEGATIONS)

def _intensity_multiplier(text: str) -> float:
    mult = 1.0
    if _contains_any(text, INTENSIFIERS): mult *= 1.35
    if _contains_any(text, DIMINISHERS): mult *= 0.75
    if '!!' in text or '???' in text: mult *= 1.15
    return mult

def _emoji_boost(text: str, acc: Dict[str, float]):
    for emo, marks in EMOJI_MAP.items():
        if _contains_any(text, marks):
            acc[emo] = acc.get(emo, 0.0) + RULE_BOOST.get(emo, 0.2)

def _contrast_dampen(text: str, acc: Dict[str, float]):
    if _contains_any(text, CONTRASTORS):
        for k in acc: acc[k] *= 0.9

def adjust_with_rules(text: str, probs: Dict[str, float]) -> Dict[str, float]:
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
                if _negated_near(t, kw): boost *= 0.35
                out[emo] = min(1.0, out.get(emo, 0.0) + boost)

    # 2) 외로움/고독 패턴 → 슬픔 크게 보강
    if any(re.search(p, t) for p in LONELY_PATTERNS):
        out['슬픔'] = min(1.0, out.get('슬픔', 0.0) + 0.25)
        # 공포 단서 거의 없으면 슬픔 추가 + 공포 약화
        if not any(h in t for h in FEAR_HINTS):
            out['슬픔'] = min(1.0, out.get('슬픔', 0.0) + 0.15)
            out['공포'] = max(0.0, out.get('공포', 0.0) - 0.10)

    # 3) 허무/상실 + 불안 공존 시 슬픔 쪽으로 가중
    if _contains_any(t, ['허망','허무','허탈','상실','공허','현타']) and _contains_any(t, ['불안','걱정','두렵','무섭']):
        out['슬픔'] = min(1.0, out.get('슬픔', 0.0) + 0.18)

    # 4) 의문/호소 패턴
    if _contains_any(t, FEAR_QUESTIONS): out['공포'] = min(1.0, out.get('공포', 0.0) + 0.12)
    if _contains_any(t, SADNESS_PHRASES): out['슬픔'] = min(1.0, out.get('슬픔', 0.0) + 0.15)

    # 5) 이모지/강도/대비 접속사
    _emoji_boost(t, out)
    mult = _intensity_multiplier(t)
    for k in out: out[k] *= mult
    _contrast_dampen(t, out)

    # 6) ㅠ/ㅜ 반복 → 슬픔 소폭 보강
    tears = len(re.findall(r'(ㅠㅠ|ㅜㅜ|ㅠ|ㅜ)', t))
    if tears >= 2: out['슬픔'] = min(1.0, out.get('슬픔', 0.0) + 0.05 * min(tears, 5))

    # 7) 샤프닝 + 정규화 + 극단치 클램프
    vec = np.array([max(1e-8, out[k]) for k in EMOS], dtype=float) ** GAMMA
    s = float(vec.sum())
    if s > 0: vec /= s
    vec = np.clip(vec, 0.0, 0.95)
    vec = vec / (vec.sum() or 1.0)

    return {EMOS[i]: float(round(vec[i], 6)) for i in range(len(EMOS))}

# ---- 모델 로드/추론
def _safe_load(model_id):
    try:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        mod = AutoModelForSequenceClassification.from_pretrained(model_id, local_files_only=True)
        print(f"[sentiment] ✅ '{model_id}' 모델 로딩 성공.")
        return tok, mod
    except Exception as e:
        print(f"[sentiment] ❌ '{model_id}' 모델 로딩 실패. 에러: {repr(e)}")
        return None, None

def init_models():
    global _e_tok, _e_mod
    if _e_tok is None:
        _e_tok, _e_mod = _safe_load(EMO_DIR)

def analyze_emotion(text: str) -> Dict:
    crisis_result = check_crisis_message(text)
    if crisis_result: return crisis_result

    contrast_result = check_contrast_emotion(text)
    if contrast_result: return contrast_result

    simple_result = check_simple_emotion(text)
    if simple_result: return simple_result

    if not _e_tok or not _e_mod or not text or not text.strip():
        return {"emotions": {e: 0.0 for e in EMOS}}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _e_mod.to(device)

    inputs = _e_tok(text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        logits = _e_mod(**inputs).logits

    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    raw_emotions = {EMOS[i]: float(probabilities[i]) for i in range(len(EMOS))}
    emotions = adjust_with_rules(text, raw_emotions)

    return {"emotions": emotions}
