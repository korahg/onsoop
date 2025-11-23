# download_model_once.py
import os
from huggingface_hub import snapshot_download

ROOT = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(ROOT, "models", "emotions_kor")  # ← 코드의 EMO_DIR과 동일
os.makedirs(TARGET, exist_ok=True)

REPO_ID = "kor-emotion-7cls"
snapshot_download(
    repo_id=REPO_ID,
    local_dir=TARGET,
    local_dir_use_symlinks=False
)
print("✅ downloaded to:", TARGET)
