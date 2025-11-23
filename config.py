# config.py
import os
from dotenv import load_dotenv
from openai import OpenAI

# .env 로드 (프로그램 시작 시 1회)
load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# OpenAI 클라이언트
client = OpenAI(api_key=OPENAI_API_KEY)

