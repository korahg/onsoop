# model/privacy.py
import os
from cryptography.fernet import Fernet


FERNET_KEY = os.getenv("FERNET_KEY")
if not FERNET_KEY:
    raise RuntimeError("FERNET_KEY missing in environment")


_f = Fernet(FERNET_KEY.encode())


def encrypt_text(plain: str) -> bytes:
    if plain is None:
        plain = ""
    return _f.encrypt(plain.encode("utf-8"))


def decrypt_text(cipher: bytes) -> str:
    return _f.decrypt(cipher).decode("utf-8")