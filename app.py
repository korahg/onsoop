# app.py
from __future__ import annotations
import os, uuid, sqlite3, threading, webbrowser, re, tempfile, json
from datetime import datetime, timedelta
from threading import Lock

from werkzeug.security import generate_password_hash, check_password_hash
from flask import (
    Flask, render_template, request, redirect, url_for,
    session as flask_session, make_response, jsonify, g
)

from dotenv import load_dotenv
load_dotenv()

# ==== 경로/환경 기본값 ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "db", "emotion_app.db")
DB_PATH = os.getenv("DB_PATH", DEFAULT_DB_PATH)
SECRET_KEY = os.getenv("SECRET_KEY", "change-this-in-production")


def _auto_migrate_columns():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    def has_col(table, name):
        cur.execute(f"PRAGMA table_info({table})")
        return any(r[1] == name for r in cur.fetchall())

    # emotion_logs
    if not has_col("emotion_logs", "session_id"):
        cur.execute("ALTER TABLE emotion_logs ADD COLUMN session_id INTEGER")
    if not has_col("emotion_logs", "created_at"):
        cur.execute("ALTER TABLE emotion_logs ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

    # conversations
    if not has_col("conversations", "session_id"):
        cur.execute("ALTER TABLE conversations ADD COLUMN session_id INTEGER")

    conn.commit()
    conn.close()

# ---- DB 테이블 보장
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      email TEXT UNIQUE,
      password_hash TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id TEXT,
      title TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversations(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id TEXT,
      role TEXT,
      encrypted_content TEXT,
      session_id INTEGER,
      timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS emotion_logs(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id TEXT,
      session_id INTEGER,
      emotions_json TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS consents(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id TEXT,
      consent INTEGER,
      version TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    conn.commit()
    conn.close()

# ---- 보안/암복호화 & 모델
from model.privacy import encrypt_text, decrypt_text
from model.sentiment import init_models, analyze_emotion
from model.chatbot import chatbot_response

# ---- Flask 앱
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = SECRET_KEY

# ==== 최초 1회 초기화 ====
_models_ready = False
_models_lock = Lock()

@app.before_request
def _maybe_warm_models():
    global _models_ready
    if not _models_ready:
        with _models_lock:
            if not _models_ready:
                try:
                    init_db()
                    init_models()
                    _models_ready = True
                    print(f"[init] DB ready @ {DB_PATH} / sentiment model loaded")
                except Exception as e:
                    print("[init error]", e)

EMOS = ['불안', '놀람', '분노', '슬픔', '중립', '행복', '혐오']

# ---- DB 유틸
def get_db():
    db = getattr(g, "db", None)
    if db is None:
        db = sqlite3.connect(DB_PATH, check_same_thread=False)
        db.row_factory = sqlite3.Row
        g.db = db
        return db
    # 이미 g.db가 있는데 닫혀 있으면 재연결
    try:
        db.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        db = sqlite3.connect(DB_PATH, check_same_thread=False)
        db.row_factory = sqlite3.Row
        g.db = db
    return db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# ---- 사용자/세션 유틸
def get_or_create_user_id() -> str:
    auth_uid = flask_session.get("auth_user_id")
    if auth_uid:
        return str(auth_uid)
    uid = flask_session.get("guest_user_id")
    if not uid:
        uid = uuid.uuid4().hex
        flask_session["guest_user_id"] = uid
    return uid

def get_current_session_id() -> int | None:
    sid = flask_session.get("current_session_id")
    return int(sid) if sid else None

from datetime import datetime

def start_new_session(user_id: str, title: str | None = None) -> int:
    conn = get_db(); cur = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(
        "INSERT INTO sessions (user_id, title, created_at) VALUES (?, ?, ?)",
        (user_id, title, now)
    )
    sid = cur.lastrowid
    conn.commit()
    # 연결은 g.db를 쓰니 여기서 닫지 않음
    flask_session["current_session_id"] = sid
    flask_session["last_activity_at"] = now
    return sid

def get_or_create_session_id(user_id: str) -> int:
    sid = get_current_session_id()
    if sid:
        return sid
    return start_new_session(user_id)

def ensure_session_by_idle_gap(user_id: str, gap_hours: int = 6):
    """동의 후 실제 저장 시점에만 호출."""
    last = flask_session.get("last_activity_at")
    try:
        last = datetime.fromisoformat(last) if last else None
    except Exception:
        last = None
    if (not last) or (datetime.now() - last > timedelta(hours=gap_hours)):
        start_new_session(user_id)
    flask_session["last_activity_at"] = datetime.now().isoformat()

# ---- 동의 상태 헬퍼
def get_consent(user_id: str) -> bool:
    if "consent" in flask_session:
        return bool(flask_session["consent"])
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        SELECT consent FROM consents
        WHERE user_id=?
        ORDER BY created_at DESC LIMIT 1
    """, (user_id,))
    row = cur.fetchone()
    ok = bool(row["consent"]) if row else False
    flask_session["consent"] = ok
    return ok

def set_consent(user_id: str, consent: bool, version: str = "v1"):
    conn = get_db(); cur = conn.cursor()
    cur.execute("INSERT INTO consents (user_id, consent, version) VALUES (?, ?, ?)",
                (user_id, 1 if consent else 0, version))
    conn.commit()
    flask_session["consent"] = bool(consent)

# ---- 대화 저장/조회
def _maybe_set_session_title(sess_id: int, text: str):
    if not text:
        return
    title = text.strip().splitlines()[0][:30]
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        UPDATE sessions
           SET title = COALESCE(NULLIF(title, ''), ?)
         WHERE id = ?
    """, (title, sess_id))
    conn.commit()  # ✨ close 하지 않음

def save_message(user_id: str, role: str, content: str):
    enc = encrypt_text(content or "")
    ensure_session_by_idle_gap(user_id, gap_hours=6)
    sess_id = get_or_create_session_id(user_id)

    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        INSERT INTO conversations (user_id, role, encrypted_content, session_id)
        VALUES (?, ?, ?, ?)
    """, (user_id, role, enc, sess_id))
    conn.commit()  # ✨ close 하지 않음

    if role == "user":
        _maybe_set_session_title(sess_id, content or "")

def load_history(user_id: str, session_id: int | None = None) -> list[dict]:
    conn = get_db(); cur = conn.cursor()
    if session_id is None:
        session_id = flask_session.get("current_session_id")
    if session_id:
        cur.execute("""
            SELECT role, encrypted_content
            FROM conversations
            WHERE user_id=? AND session_id=?
            ORDER BY timestamp ASC
        """, (user_id, session_id))
    else:
        cur.execute("""
            SELECT role, encrypted_content
            FROM conversations
            WHERE user_id=?
            ORDER BY timestamp ASC
        """, (user_id,))
    rows = cur.fetchall()
    hist = []
    for r in rows:
        try:
            hist.append({"role": r["role"], "content": decrypt_text(r["encrypted_content"])})
        except Exception:
            hist.append({"role": r["role"], "content": "[복호화 실패]"})
    return hist

def save_emotion_log(user_id: str, emotions: dict):
    ensure_session_by_idle_gap(user_id, gap_hours=6)
    sess_id = get_or_create_session_id(user_id)
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        INSERT INTO emotion_logs (user_id, session_id, emotions_json)
        VALUES (?, ?, ?)
    """, (user_id, sess_id, json.dumps(emotions, ensure_ascii=False)))
    conn.commit()

# ---- 라우트: 홈/챗봇
@app.route("/")
def index():
    return redirect(url_for("chatbot"))

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    user_id = get_or_create_user_id()
    consent_ok = get_consent(user_id)

    if request.method == "POST":
        user_message = (request.form.get("message") or "").strip()

        # 동의 안 했으면 저장/응답 금지
        if not consent_ok:
            return redirect(url_for("chatbot", need_consent=1))

        if user_message:
            flask_session["last_user_text"] = user_message

            history = load_history(user_id)
            try:
                analysis = analyze_emotion(user_message) or {}
            except Exception as e:
                analysis = {}
                print("[analyze_emotion error]", e)

            try:
                bot_reply = chatbot_response(user_message, history=history, context=analysis)
            except Exception as e:
                bot_reply = f"죄송해요. 응답 생성 중 오류가 발생했어요: {e}"

            # 동의한 경우에만 영구 저장 (세션 생성/갱신)
            save_message(user_id, "user", user_message)
            save_message(user_id, "assistant", bot_reply)
            if analysis and isinstance(analysis, dict):
                payload = analysis.get("emotions") or analysis
                try:
                    save_emotion_log(user_id, payload)
                except Exception as e:
                    print("[save_emotion_log warn]", e)

        return redirect(url_for("chatbot"))

    # GET: 동의 후에만 히스토리 표시
    history = load_history(user_id) if consent_ok else []
    return make_response(render_template("chatbot.html",
                                         history=history,
                                         consent_ok=consent_ok,
                                         need_consent=request.args.get("need_consent") == "1"))

@app.route("/new_chat")
def new_chat():
    user_id = get_or_create_user_id()
    if get_consent(user_id):
        start_new_session(user_id)
    return redirect(url_for("chatbot"))

# ====== [여기 추가] 실시간 분포 후처리/가중치 헬퍼 ======
def _apply_length_weight(w: float, txt: str) -> float:
    """짧은 문장 과대반영 방지: 0.5~1.0 가중."""
    L = max(1, min(40, len((txt or '').strip())))   # 1~40자로 클램프
    len_w = 0.5 + 0.5 * (L / 40.0)                  # 0.5 ~ 1.0
    return w * len_w

def _postprocess_distribution(acc: dict[str, float]) -> dict[str, float]:
    """바닥값 → 정규화 → 과신 캡(Top1-Top2 격차 작을 때 평탄화)."""
    for k in acc:
        acc[k] = max(float(acc[k] or 0.0), 1e-6)
    s = sum(acc.values()) or 1.0
    acc = {k: v / s for k, v in acc.items()}
    vals = sorted(acc.values(), reverse=True)
    if len(vals) >= 2 and (vals[0] - vals[1]) < 0.25:
        for k in acc:
            acc[k] *= 0.9
        s = sum(acc.values()) or 1.0
        acc = {k: v / s for k, v in acc.items()}
    return acc
# =====================================================

# ---- 실시간 감정 분포(JSON) + 스티커 추천
WEIGHTS = [0.6, 0.3, 0.1]  # 최근 → 과거

@app.route("/emotion_mix")
def emotion_mix():
    # 프론트·모델 혼용 대비: 여기서는 '공포' 라벨을 표준으로 사용
    emos = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
    user_id = get_or_create_user_id()
    session_id = flask_session.get("current_session_id")

    if not get_consent(user_id) or not session_id:
        return jsonify({"emotions": {e: 0.0 for e in emos}, "sticker": None})

    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        SELECT encrypted_content
        FROM conversations
        WHERE user_id=? AND session_id=? AND role='user'
        ORDER BY timestamp DESC
        LIMIT 3
    """, (user_id, session_id))
    rows = cur.fetchall()

    if not rows:
        return jsonify({"emotions": {e: 0.0 for e in emos}, "sticker": None})

    acc = {e: 0.0 for e in emos}
    latest_txt = ""

    for i, r in enumerate(rows):
        try:
            txt = decrypt_text(r["encrypted_content"])
        except Exception:
            txt = ""
        if i == 0:
            latest_txt = txt

        try:
            res = analyze_emotion(txt) or {}
            dist = (res.get("emotions") or res) if isinstance(res, dict) else {}
        except Exception:
            dist = {}

        # 기본 가중 + 문장 길이 가중치
        w = WEIGHTS[i] if i < len(WEIGHTS) else 0.05
        w = _apply_length_weight(w, txt)

        # 누적 (라벨 혼용 대비: '불안'이 올 경우 '공포'로 합산)
        for k in emos:
            if k == '공포':
                v = float(dist.get('공포', dist.get('불안', 0.0)) or 0.0)
            else:
                v = float(dist.get(k, 0.0) or 0.0)
            acc[k] += w * v

    # ----- 로그 EMA(최신 10개)와 50:50 믹스 -----
    cur.execute("""
        SELECT emotions_json
        FROM emotion_logs
        WHERE user_id=? AND session_id=?
        ORDER BY created_at DESC
        LIMIT 10
    """, (user_id, session_id))
    logs = cur.fetchall()
    if logs:
        alpha = 0.6
        ema = {e: 0.0 for e in emos}
        first = True
        for r in logs:
            raw = r["emotions_json"]
            try:
                d = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except Exception:
                try:
                    d = eval(raw) if isinstance(raw, str) else {}
                except Exception:
                    d = {}
            curv = {
                '공포': float(d.get('공포', d.get('불안', 0.0)) or 0.0),
                '놀람': float(d.get('놀람', 0.0) or 0.0),
                '분노': float(d.get('분노', 0.0) or 0.0),
                '슬픔': float(d.get('슬픔', 0.0) or 0.0),
                '중립': float(d.get('중립', 0.0) or 0.0),
                '행복': float(d.get('행복', 0.0) or 0.0),
                '혐오': float(d.get('혐오', 0.0) or 0.0),
            }
            if first:
                ema = curv; first = False
            else:
                for k in emos:
                    ema[k] = alpha * curv[k] + (1 - alpha) * ema[k]

        for k in emos:
            acc[k] = 0.5 * acc[k] + 0.5 * ema[k]

    # ----- 극단값 완화/정규화/과신 캡 -----
    acc = _postprocess_distribution(acc)

    # ----- 간단 스티커 추천 -----
    sticker = None
    if latest_txt:
        last = analyze_emotion(latest_txt) or {}
        last_dist = (last.get("emotions") or last) if isinstance(last, dict) else {}
        sad   = float(last_dist.get('슬픔', 0.0) or 0.0)
        happy = float(last_dist.get('행복', 0.0) or 0.0)
        if sad >= 0.55 and sad >= happy + 0.10:
            sticker = {"type": "cheer", "text": "힘내요!"}
        elif happy >= 0.60 and happy >= sad + 0.10:
            sticker = {"type": "congrats", "text": "축하해요!"}

    return jsonify({"emotions": acc, "sticker": sticker})

# ---- 동의 수락/철회
@app.route("/consent", methods=["POST"])
def consent_accept():
    user_id = get_or_create_user_id()
    agree = request.form.get("agree") == "on"
    set_consent(user_id, agree, version="v1")
    flask_session.pop("current_session_id", None)
    flask_session.pop("last_user_text", None)
    if agree:
        start_new_session(user_id)
    return redirect(url_for("chatbot"))

@app.route("/consent/revoke", methods=["POST"])
def consent_revoke():
    user_id = get_or_create_user_id()
    set_consent(user_id, False, version="v1")
    flask_session.pop("current_session_id", None)
    flask_session.pop("last_user_text", None)
    return redirect(url_for("chatbot"))

# ---- 회원/인증
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = (request.form.get("password") or "").strip()
        if not email or not password:
            return render_template("register.html", error="이메일/비밀번호를 입력해 주세요.")

        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE email=?", (email,))
        if cur.fetchone():
            return render_template("register.html", error="이미 가입된 이메일입니다.")

        ph = generate_password_hash(password)
        cur.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)", (email, ph))
        conn.commit()

        flask_session["auth_user_email"] = email
        flask_session["auth_user_id"] = cur.lastrowid
        get_or_create_user_id()
        return redirect(url_for("chatbot"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = (request.form.get("password") or "").strip()

        conn = get_db(); cur = conn.cursor()
        cur.execute("SELECT id, password_hash FROM users WHERE email=?", (email,))
        row = cur.fetchone()
        if (not row) or (not check_password_hash(row["password_hash"], password)):
            return render_template("login.html", error="이메일 또는 비밀번호가 올바르지 않습니다.")

        flask_session["auth_user_email"] = email
        flask_session["auth_user_id"] = row["id"]
        get_or_create_user_id()
        return redirect(url_for("chatbot"))

    return render_template("login.html")

@app.route("/logout")
def logout():
    flask_session.pop("auth_user_email", None)
    flask_session.pop("auth_user_id", None)
    return redirect(url_for("chatbot"))

# ---- 상담 내용 요약 API
try:
    from config import client, OPENAI_MODEL
except Exception:
    client = None
    OPENAI_MODEL = None

def _clean_markups(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\*\*(.*?)\*\*", r"\1", s, flags=re.S)
    s = re.sub(r"__(.*?)__", r"\1", s, flags=re.S)
    s = re.sub(r"`(.*?)`", r"\1", s, flags=re.S)
    return s.strip()

@app.route("/analyze_session", methods=["POST"])
def analyze_session():
    user_id = get_or_create_user_id()
    history = load_history(user_id)
    N = 40
    msgs = history[-N:]

    if client and OPENAI_MODEL:
        prompt = (
            "[지시사항]\n"
            "당신은 심리 상담 전문가입니다. 아래 상담 대화를 바탕으로 다음 항목을 한국어 '평문'으로만 작성하세요.\n"
            "마크다운 금지. 제목은 번호와 마침표로만 표기.\n\n"
            "1. 핵심 문제 요약 (3-4문장)\n"
            "2. 주요 감정 분석 (근거 포함)\n"
            "3. 현재 심리 상태 진단\n"
            "4. 실천 가능한 조언/안내"
        )
        messages = [{"role":"system","content":"너는 공감하는 상담 비서다. 마크다운 없이 요약하라."}]
        for m in msgs:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role":"user","content": prompt})

        try:
            chat = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.2,
                timeout=30,
            )
            summary = (chat.choices[0].message.content or "").strip()
            summary = _clean_markups(summary)
            return jsonify({"log_count": len(msgs), "analysis": summary})
        except Exception as e:
            print("[analyze_session warn]", e)

    joined = "\n".join([f'[{m["role"]}] {m["content"]}' for m in msgs])
    fallback = "1. 최근 대화 핵심 요약\n" + joined[:1200]
    fallback = _clean_markups(fallback)
    return jsonify({"log_count": len(msgs), "analysis": fallback})

# ---- STT (Whisper 폴백)
_STT_READY = False
_whisper_model = None
_whisper_err = None

def _init_stt_if_needed():
    global _STT_READY, _whisper_model, _whisper_err
    if _STT_READY:
        return
    try:
        import whisper  # pip install openai-whisper
        model_name = os.getenv("WHISPER_MODEL", "base")
        _whisper_model = whisper.load_model(model_name)  # ffmpeg 필요
        _STT_READY = True
        print(f"[STT] Whisper ready: {model_name}")
    except Exception as e:
        _whisper_err = e
        _STT_READY = True
        print("[STT] Whisper not available:", repr(e))

@app.route("/stt", methods=["POST"])
def stt_transcribe():
    user_id = get_or_create_user_id()
    if not get_consent(user_id):
        return jsonify({"ok": False, "error": "consent_required"}), 403

    _init_stt_if_needed()
    if _whisper_model is None:
        return jsonify({"ok": False, "error": f"stt_unavailable: {_whisper_err}"}), 501

    file = request.files.get("audio")
    if not file:
        return jsonify({"ok": False, "error": "no_audio"}), 400

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    try:
        file.save(tmp.name)
        result = _whisper_model.transcribe(tmp.name, language="ko", fp16=False)
        text = (result.get("text") or "").strip()
        return jsonify({"ok": True, "text": text})
    except Exception as e:
        return jsonify({"ok": False, "error": repr(e)}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

# ---- 리포트/기록/세션 전환/리셋
@app.route("/report")
def report():
    user_id = get_or_create_user_id()
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        SELECT created_at, emotions_json
        FROM emotion_logs
        WHERE user_id=?
        ORDER BY created_at ASC
    """, (user_id,))
    rows = cur.fetchall()

    labels = []
    series = {k: [] for k in ['놀람','불안','혐오','중립','행복','분노','슬픔']}
    for r in rows:
        labels.append(r["created_at"])
        try:
            data = json.loads(r["emotions_json"]) if isinstance(r["emotions_json"], str) else (r["emotions_json"] or {})
        except Exception:
            try:
                data = eval(r["emotions_json"]) if isinstance(r["emotions_json"], str) else {}
            except Exception:
                data = {}
        for k in series.keys():
            if k == '불안':
                v = float((data.get('불안', data.get('공포', 0.0))) or 0.0)
            else:
                v = float(data.get(k, 0.0) or 0.0)
            series[k].append(round(v, 4))
    datasets = [{"label": k, "data": series[k], "tension": 0.2} for k in series.keys()]
    chart_data = {"labels": labels, "datasets": datasets}
    return render_template("report.html", chart_data=json.dumps(chart_data, ensure_ascii=False))


def load_messages_for_session(user_id: str, session_id: int) -> list[dict]:
    """특정 회차의 대화를 복호화하여 UI에 넘길 형태로 반환."""
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        SELECT role, encrypted_content, timestamp
        FROM conversations
        WHERE user_id=? AND session_id=?
        ORDER BY timestamp ASC
    """, (user_id, session_id))
    rows = cur.fetchall()
    msgs = []
    for r in rows:
        try:
            msgs.append({
                "role": r["role"],
                "content": decrypt_text(r["encrypted_content"]),
                "timestamp": r["timestamp"],
            })
        except Exception:
            msgs.append({
                "role": r["role"],
                "content": "[복호화 실패]",
                "timestamp": r["timestamp"],
            })
    return msgs

@app.route("/history", endpoint="history")
def history_page():
    user_id = get_or_create_user_id()
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        SELECT
            s.id            AS id,
            s.title         AS title,
            s.created_at    AS created_at,
            COUNT(c.id)     AS msg_count
        FROM sessions s
        LEFT JOIN conversations c
               ON c.session_id = s.id AND c.user_id = s.user_id
        WHERE s.user_id = ?
        GROUP BY s.id, s.title, s.created_at
        HAVING COUNT(c.id) > 0
        ORDER BY s.created_at DESC
    """, (user_id,))
    rows = cur.fetchall()

    sessions = []
    for r in rows:
        sessions.append({
            "id": r["id"],
            "title": r["title"] or "회차",
            "created_at": r["created_at"] or "",
            "msg_count": r["msg_count"],
        })
    return render_template("history.html", sessions=sessions)

@app.route("/history/<int:sid>", endpoint="history_detail")
def history_detail(sid: int):
    user_id = get_or_create_user_id()

    # 회차 소유권 확인
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT id, title, created_at FROM sessions WHERE id=? AND user_id=?", (sid, user_id))
    sess = cur.fetchone()
    if not sess:
        conn.close()
        # 남의 회차이거나 없는 회차면 목록으로
        return redirect(url_for("history"))

    # 해당 회차의 메시지 로드
    cur.execute("""
        SELECT role, encrypted_content, timestamp
        FROM conversations
        WHERE user_id=? AND session_id=?
        ORDER BY timestamp ASC, id ASC
    """, (user_id, sid))
    rows = cur.fetchall()
    conn.close()

    messages = []
    for r in rows:
        try:
            messages.append({
                "role": r["role"],
                "content": decrypt_text(r["encrypted_content"]),
                "timestamp": r["timestamp"]
            })
        except Exception:
            messages.append({
                "role": r["role"],
                "content": "[복호화 실패]",
                "timestamp": r["timestamp"]
            })

    # history.html이 회차 상세도 렌더링할 수 있도록 다음 변수들을 넘겨줍니다.
    return render_template(
        "history.html",
        sessions=None,            # 목록 뷰가 아니라는 신호
        selected_session=sess,    # 회차 메타
        messages=messages         # 복호화된 메시지 배열
    )

# ====== 상담 회차 삭제 기능 ======
@app.route("/history/<int:sid>/delete", methods=["POST"], endpoint="history_delete")
def history_delete(sid: int):
    """특정 상담 회차 + 그 회차의 메시지/감정로그를 삭제."""
    user_id = get_or_create_user_id()
    conn = get_db(); cur = conn.cursor()

    # 내 회차인지 확인 (남의 회차 삭제 방지)
    cur.execute("SELECT id FROM sessions WHERE id=? AND user_id=?", (sid, user_id))
    row = cur.fetchone()
    if not row:
        return redirect(url_for("history"))

    # 1) 이 회차의 대화/감정 로그 삭제
    cur.execute("DELETE FROM conversations WHERE user_id=? AND session_id=?", (user_id, sid))
    cur.execute("DELETE FROM emotion_logs WHERE user_id=? AND session_id=?", (user_id, sid))

    # 2) 세션(회차) 자체 삭제
    cur.execute("DELETE FROM sessions WHERE id=? AND user_id=?", (sid, user_id))
    conn.commit()

    # 3) 현재 사용 중인 회차를 지운 경우 세션에서 제거
    if flask_session.get("current_session_id") == sid:
        flask_session.pop("current_session_id", None)

    return redirect(url_for("history"))

@app.route("/switch_session/<int:sid>")
def switch_session(sid: int):
    user_id = get_or_create_user_id()
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT id FROM sessions WHERE id=? AND user_id=?", (sid, user_id))
    row = cur.fetchone()
    if not row:
        if get_consent(user_id):
            start_new_session(user_id)
    else:
        flask_session["current_session_id"] = sid
    flask_session.pop("last_user_text", None)
    return redirect(url_for("chatbot"))

@app.route("/reset")
def reset_chat():
    user_id = get_or_create_user_id()
    flask_session.pop("current_session_id", None)
    flask_session.pop("last_user_text", None)
    if get_consent(user_id):
        start_new_session(user_id)
    return redirect(url_for("chatbot"))

# ---- 개발용: 자동 브라우저 열기
def _open_browser():
    # 서버는 0.0.0.0에 바인드해도, 브라우저는 로컬루프백으로 접속!
    port = int(os.getenv("PORT", 5000))
    webbrowser.open_new(f"http://127.0.0.1:{port}/chatbot")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
