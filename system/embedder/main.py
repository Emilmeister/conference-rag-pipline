from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import uvicorn
import os

app = FastAPI()

EMB_MODEL = os.getenv("EMB_MODEL", 'sentence-transformers/sentence-t5-xl')

model = SentenceTransformer(EMB_MODEL)


class TextRequest(BaseModel):
    text: str


@app.post("/embedding")
def get_embedding(request: TextRequest):
    # Получение эмбеддингов
    with torch.no_grad():
        embeddings = model.encode([request.text])
    print(request.text, embeddings.tolist()[0])
    return {"embedding": embeddings.tolist()[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
