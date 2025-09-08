import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# .env 선로딩
load_dotenv(override=True)

from .schemas import (
    EmbedRequest, EmbedResponse,
    SimilarityRequest, SimilarityResponse,
    ClassifyRequest, ClassifyResponse,
)
from .services import ModelService

app = FastAPI(title="KoBERT FastAPI", version="0.1.0")

# 서비스 인스턴스 (환경변수 반영 후 생성)
svc = ModelService()

# CORS
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": svc.mode,
        "model": os.getenv("MODEL_NAME_OR_PATH"),
        "max_length": os.getenv("MAX_LENGTH"),
        "device_env": os.getenv("DEVICE"),
    }

@app.get("/gpu")
def gpu():
    try:
        import torch
        ok = torch.cuda.is_available()
        return {
            "cuda_available": ok,
            "device": torch.cuda.get_device_name(0) if ok else "cpu",
        }
    except Exception as e:
        return {"cuda_available": False, "error": str(e)}

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    try:
        vecs = svc.embed(req.texts)
        return {"embeddings": vecs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similarity", response_model=SimilarityResponse)
def similarity(req: SimilarityRequest):
    try:
        score = svc.similarity(req.a, req.b)
        return {"score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    try:
        labels, scores = svc.classify(req.texts, return_probs=req.return_probs)
        return {"labels": labels, "scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
