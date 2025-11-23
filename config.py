import os
from dotenv import load_dotenv
from openai import OpenAI

# .env 로드 (로컬 개발용)
# override=False 로 두면, Render에서 설정한 환경변수를 .env가 덮어쓰지 않음
load_dotenv(override=False)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# 🔍 디버그용 로그 (키 값은 찍지 않고, 존재 여부/이름만 확인)
print("DEBUG:: HAS_OPENAI_API_KEY =", bool(OPENAI_API_KEY))
print("DEBUG:: ENV OPENAI KEYS =", [k for k in os.environ.keys() if "OPENAI" in k])

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY가 설정되지 않았습니다. Render Environment 탭에서 확인하세요.")

client = OpenAI(api_key=OPENAI_API_KEY)
