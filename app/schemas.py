from pydantic import BaseModel, Field
from typing import List

class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1)

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

class SimilarityRequest(BaseModel):
    a: str
    b: str

class SimilarityResponse(BaseModel):
    score: float  # cosine similarity

class ClassifyRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1)
    return_probs: bool = True

class ClassifyResponse(BaseModel):
    labels: List[str]
    scores: List[List[float]]  # probs or logits
