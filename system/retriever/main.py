import uuid
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from agents import Agent
from agents import set_default_openai_client, set_default_openai_api, set_trace_processors, Runner, trace
from phoenix.otel import register
from openai.types.responses import EasyInputMessageParam
from dotenv import load_dotenv
from pymilvus import MilvusClient
from opensearchpy import OpenSearch
from agents.models import openai_provider
from openai import AsyncOpenAI
import logging
import requests
import numpy as np
import os
import ssl

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

ssl._create_default_https_context = ssl._create_unverified_context

PHOENIX_TRACE_URL = os.getenv("PHOENIX_TRACE_URL")
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")

# configure the Phoenix tracer
set_trace_processors([])
tracer_provider = register(
    project_name=PHOENIX_PROJECT_NAME,  # Default is 'default'
    endpoint=PHOENIX_TRACE_URL,
    auto_instrument=True
)

set_default_openai_client(AsyncOpenAI(base_url=OPENAI_API_URL, api_key=OPENAI_API_KEY, timeout=60 * 5))
set_default_openai_api('chat_completions')
openai_provider.DEFAULT_MODEL = DEFAULT_MODEL

MILVUS_URI = os.getenv('MILVUS_URI')
OPENSEARCH_HOST = os.getenv('OPENSEARCH_HOST')
OPENSEARCH_PORT = int(os.getenv('OPENSEARCH_PORT'))
OPENSEARCH_USER = os.getenv('OPENSEARCH_USER')
OPENSEARCH_PASS = os.getenv('OPENSEARCH_PASS')
MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME')
OPENSEARCH_INDEX_NAME = os.getenv('OPENSEARCH_INDEX_NAME')
EMB_DIM = int(os.getenv('EMB_DIM'))
EMB_URL = os.getenv('EMB_URL')
RERANKER_URL = os.getenv('RERANKER_URL')

milvus_client = MilvusClient(MILVUS_URI)

opensearch_client = OpenSearch(
    hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASS),
    http_compress=True,
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False
)
app = FastAPI()


class RetrieveRequest(BaseModel):
    query: str
    classic_search_num: int
    vector_search_num: int
    max_get_num: int


class HypotheticalDocument(BaseModel):
    reasoning: str
    hypothetical_document_text: str


class MultipleQueries(BaseModel):
    reasoning: str
    multiple_queries: List[str]


hypo_docs_generator = Agent(
    name="Hypo documents generator",
    instructions="""
Пользователь использует RAG систему для получения релевантных документов по вопросу.
По переданному вопросу ты должен сгенерировать потенциальный документ, который может находиться в базе данных.
Его будут использовать для нахождения реального документа через косинусное расстояние эмбеддингов.
    """,
    output_type=HypotheticalDocument
)

multi_query_generator = Agent(
    name="Multi Query generator",
    instructions="""
Пользователь использует RAG систему для получения релевантных документов.
Твоя задача по вопросу пользователя сгенерировать 2 потенциальных вопросов которые 
в зависимости от изначального вопроса помогут уточнить его или более широко его раскрыть.
    """,
    output_type=MultipleQueries
)


async def get_multiple_queries(query: str) -> List[str]:
    result = await Runner.run(multi_query_generator, [EasyInputMessageParam(role='user', content=query)])
    return MultipleQueries.model_validate(result.final_output).multiple_queries


async def get_hypo_documents(query: str) -> str:
    result = await Runner.run(hypo_docs_generator, [EasyInputMessageParam(role='user', content=query)])
    return HypotheticalDocument.model_validate(result.final_output).hypothetical_document_text


def get_embedding(text: str) -> List[int]:
    return requests.post(url=EMB_URL, headers={"Content-Type": "application/json"}, json={"text": text}).json().get(
        'embedding')


def similar_docs_milvus(query, n=5):
    vector = get_embedding(query)

    results = milvus_client.search(
        collection_name=MILVUS_COLLECTION_NAME,
        data=[vector],
        limit=n,
        output_fields=['parent_id', 'room_uuid', 'base_chunk', 'parent_chunk']
    )
    answer = []
    for result in results[0]:
        d = dict()
        d['id'] = result['id']
        entity = result['entity']
        d['parent_id'] = entity['parent_id']
        d['room_uuid'] = entity['room_uuid']
        d['base_chunk'] = entity['base_chunk']
        d['parent_chunk'] = entity['parent_chunk']
        d['text'] = entity['base_chunk']
        answer.append(d)
    return answer


def similar_docs_opensearch(query, n=5):
    body = {
        'size': n,
        'query': {
            'multi_match': {
                'query': query,
                'fields': ['base_chunk']
            }
        }
    }
    answer = []
    results = opensearch_client.search(
        body=body,
        index=OPENSEARCH_INDEX_NAME
    )['hits']['hits']
    for result in results:
        d = dict()
        d['id'] = result['_id']
        source = result['_source']
        d['parent_id'] = source['parent_id']
        d['room_uuid'] = source['room_uuid']
        d['base_chunk'] = source['base_chunk']
        d['parent_chunk'] = source['parent_chunk']
        d['text'] = source['base_chunk']
        d['score'] = result['_score']
        answer.append(d)
    return answer


def get_reranked_docs(query: str, texts: List[dict]):
    req = {
        "query": query,
        "texts": texts
    }
    return requests.post(RERANKER_URL, json=req).json()


@app.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    with trace(workflow_name="video-conference-retriever", group_id=str(uuid.uuid4())):
        multiple_queries = await get_multiple_queries(request.query)
        hypo_docs = [await get_hypo_documents(x) for x in multiple_queries]
        docs = []
        for hypo_doc in hypo_docs:
            docs.extend(similar_docs_milvus(hypo_doc, n=request.vector_search_num))
            docs.extend(similar_docs_opensearch(hypo_doc, n=request.classic_search_num))

        docs = list({x['id']: x for x in docs}.values())
        reranked_docs = get_reranked_docs(request.query, docs)

        parent_max_score_dict = dict()

        for reranked_doc in reranked_docs['texts']:
            if reranked_doc['parent_id'] in parent_max_score_dict:
                if parent_max_score_dict[reranked_doc['parent_id']] < reranked_doc['score']:
                    parent_max_score_dict[reranked_doc['parent_id']] = reranked_doc['score']
            else:
                parent_max_score_dict[reranked_doc['parent_id']] = reranked_doc['score']

        reranked_docs_set = []

        for reranked_doc in reranked_docs['texts']:
            if parent_max_score_dict[reranked_doc['parent_id']] == reranked_doc['score']:
                reranked_docs_set.append(reranked_doc)

        reranked_scores = [x['score'] for x in reranked_docs_set]

        idx_s = np.argsort(-np.array(list(reranked_scores)))
        idx_s = idx_s[:min(len(idx_s), request.max_get_num)]

        texts_sort = []
        for idx in idx_s:
            texts_sort = texts_sort + [reranked_docs_set[idx]]

        return texts_sort


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
