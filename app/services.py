from __future__ import annotations

import os
import json
import numpy as np


def _env_stripped(key: str) -> str | None:
    v = os.getenv(key)
    if v is None:
        return None
    return v.split("#", 1)[0].strip() or None

def _env_int(key: str) -> int | None:
    v = _env_stripped(key)
    if not v:
        return None
    try:
        return int(v)
    except Exception:
        return None

def _env_json(key: str):
    v = _env_stripped(key)
    if not v:
        return None
    try:
        return json.loads(v)
    except Exception:
        return None


class ModelService:
    def __init__(self):
        # 순환 방지: 이곳에서 지연 import
        from .model import KoBERTEmbedding, KoBERTClassifier  # noqa: WPS433

        name = _env_stripped("MODEL_NAME_OR_PATH") or "skt/kobert-base-v1"
        ckpt = _env_stripped("FINETUNED_CKPT") or ""
        self.labels = _env_json("LABELS_JSON")
        num_labels = _env_int("NUM_LABELS")

        if ckpt:
            # 분류 모드
            self.clf = KoBERTClassifier(ckpt_or_name=ckpt, num_labels=num_labels, labels=self.labels)
            self.emb = None
            self.mode = "classification"
        else:
            # 임베딩 모드
            self.emb = KoBERTEmbedding(model_name_or_path=name)
            self.clf = None
            self.mode = "embedding"

        print(f"[ModelService] mode={self.mode} model={name} ckpt={ckpt or '-'}")

    def embed(self, texts):
        if self.emb is None:
            raise RuntimeError("Server is in classification mode; embedding disabled.")
        return self.emb.embed(texts)

    def classify(self, texts, return_probs: bool = True):
        if self.clf is None:
            raise RuntimeError("Server is in embedding mode; classification disabled.")
        scores = self.clf.predict(texts, return_probs=return_probs)
        return (self.labels or []), scores

    def similarity(self, a: str, b: str) -> float:
        """코사인 유사도"""
        vecs = self.embed([a, b])
        v1 = np.asarray(vecs[0], dtype=np.float32)
        v2 = np.asarray(vecs[1], dtype=np.float32)
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0.0:
            return 0.0
        return float(np.dot(v1, v2) / denom)
