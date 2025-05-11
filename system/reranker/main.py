from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import torch
import uvicorn

app = FastAPI()

cross_encoder = HuggingFaceCrossEncoder(
    model_name='amberoad/bert-multilingual-passage-reranking-msmarco',
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
)

tokenizer = cross_encoder.client.tokenizer
max_seq_len = (cross_encoder.client.model.bert.embeddings.position_embeddings.num_embeddings - 5) // 2


class RerankRequest(BaseModel):
    query: str
    texts: List[dict]


@app.post("/rerank")
def get_embedding(request: RerankRequest) -> RerankRequest:
    # Получение эмбеддингов

    if len(request.texts) == 0:
        return request

    input_ids_list = [tokenizer.encode(x['text']) for x in request.texts]
    input_ids_query = tokenizer.encode(request.query)

    input_ids_list = [x[:min(len(x), max_seq_len)] for x in input_ids_list]
    input_ids_query = input_ids_query[:min(len(input_ids_query), max_seq_len)]

    cut_query = tokenizer.decode(input_ids_query)

    scores = list(cross_encoder.score([(cut_query, tokenizer.decode(input_ids)) for input_ids in input_ids_list]))

    for i, text in enumerate(request.texts):
        text['score'] = float(scores[i])

    return request


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
