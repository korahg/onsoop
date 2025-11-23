# model/chatbot.py
from config import client, OPENAI_MODEL
import traceback

CHATBOT_SYSTEM = """너는 한국어 심리·정서 지원 챗봇 ‘온숲’이야.  
너의 역할은 사용자가 마음을 편하게 털어놓을 수 있도록 안전한 공간을 제공하고,  
섬세하게 감정을 읽어주며, 따뜻하고 전문적인 태도로 위로와 실천 가능한 조언을 건네는 거야.  

원칙:  
1. 사용자의 감정을 존중하고 따뜻하게 공감해줄 것.  
2. 답변은 간결하고 명확하게, 보통 2~4문장 정도로 할 것.  
3. 사용자가 더 깊이 이야기할 수 있도록 부드럽고 열린 질문을 던질 것.  
4. 감정 상태에 맞춘 현실적이고 작은 실행 지침을 제시할 것.  
5. 의학적 진단은 하지 말고, 필요 시 전문 상담이나 의료 도움을 자연스럽게 권할 것.
"""

def chatbot_response(user_msg: str, history=None, context: dict = None) -> str:
    if not user_msg or not user_msg.strip():
        return "어떤 마음이 드시는지 편하게 적어주세요."

    # 1) OpenAI로 보낼 messages 구성
    messages = [{"role": "system", "content": CHATBOT_SYSTEM}]

    # ▼▼▼▼▼ 컨텍스트를 시스템 메시지 다음에 추가합니다 ▼▼▼▼▼
    if context:
        # 주요 감정 찾기 (가장 점수가 높은 감정)
        primary_emotion = "특정되지 않음"
        if context.get("emotions"):
            primary_emotion = max(context["emotions"], key=context["emotions"].get)

        context_str = (
            f"[Internal Analysis]\n"
            f"- User's overall sentiment: {context.get('sentiment', 'N/A')}\n"
            f"- Primary detected emotion: {primary_emotion}\n"
            f"- Emotion scores: {context.get('emotions', 'N/A')}\n"
            f"Based on this analysis, provide a tailored, empathetic response."
        )
        # 시스템 메시지와 사용자 메시지 사이에 컨텍스트를 추가하여 AI에게 내부 정보를 제공
        messages.append({"role": "system", "content": context_str})
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    if history:
        for m in history:
            if m["role"] in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": user_msg.strip()})


    try:
        chat = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            timeout=30,
        )
        return (chat.choices[0].message.content or "").strip() or "말씀 감사합니다. 조금 더 이야기해도 괜찮을까요?"

    except Exception as e1:
        try:
            r = client.responses.create(
                model=OPENAI_MODEL,
                input=messages,   # responses API도 대화목록 그대로 전달
                timeout=30,
            )
            txt = ""
            if hasattr(r, "output") and r.output:
                for item in r.output:
                    if getattr(item, "type", "") == "output_text":
                        txt += getattr(item, "content", "")
            if not txt and hasattr(r, "output_text"):
                txt = r.output_text
            return (txt or "말씀 감사합니다. 조금 더 이야기해도 괜찮을까요?").strip()

        except Exception as e2:
            print("[chatbot_response ERROR 1st try]", repr(e1))
            print("[chatbot_response ERROR 2nd try]", repr(e2))
            traceback.print_exc()
            return "지금은 답변을 만들기 어려워요. (힌트: 키/모델 권한 또는 네트워크)"