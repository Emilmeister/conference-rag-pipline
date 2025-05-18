from typing import List

from opensearchpy import OpenSearch
from kafka import KafkaConsumer
from pymilvus import MilvusClient
from pymilvus import DataType
import os
from dotenv import load_dotenv
from pymilvus import utility
import uuid
import json
import logging
import requests
from pymilvus.orm.connections import Connections

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

MILVUS_URI = os.getenv('MILVUS_URI')
OPENSEARCH_HOST = os.getenv('OPENSEARCH_HOST')
OPENSEARCH_PORT = int(os.getenv('OPENSEARCH_PORT'))
OPENSEARCH_USER = os.getenv('OPENSEARCH_USER')
OPENSEARCH_PASS = os.getenv('OPENSEARCH_PASS')
MILVUS_COLLECTION_NAME = os.getenv('MILVUS_COLLECTION_NAME')
OPENSEARCH_INDEX_NAME = os.getenv('OPENSEARCH_INDEX_NAME')
EMB_DIM = int(os.getenv('EMB_DIM'))
EMB_URL = os.getenv('EMB_URL')
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
DB_SAVER_INPUT_TOPIC = os.getenv('DB_SAVER_INPUT_TOPIC')
DB_SAVER_KAFKA_GROUP_ID = os.getenv('DB_SAVER_KAFKA_GROUP_ID')

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

if not utility.has_collection(MILVUS_COLLECTION_NAME, using=Connections().list_connections()[-1][0]):
    milvus_client.create_collection(
        collection_name=MILVUS_COLLECTION_NAME,
        dimension=EMB_DIM,
        primary_field_name="id",
        id_type=DataType.VARCHAR,
        vector_field_name="vector",
        metric_type="COSINE",
        auto_id=False,
        max_length=65535
    )

if not opensearch_client.indices.exists(index=OPENSEARCH_INDEX_NAME):
    index_body = {
        'settings': {
            'index': {
                'number_of_shards': 4
            }
        }
    }
    response = opensearch_client.indices.create(OPENSEARCH_INDEX_NAME, body=index_body)


def get_embedding(text: str) -> List[int]:
    return requests.post(url=EMB_URL, headers={"Content-Type": "application/json"}, json={"text": text}).json().get('embedding')


# Инициализация Kafka Consumer и Producer
consumer = KafkaConsumer(
    DB_SAVER_INPUT_TOPIC,
    group_id=DB_SAVER_KAFKA_GROUP_ID,
    bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Прослушивание топика входящих сообщений
for message in consumer:
    try:
        logging.info('Received message: %s', message)
        room_uuid = message.value.get('room_uuid')
        large_chunks = message.value.get('chunks')
        for large_chunk in large_chunks:
            parent_id = str(uuid.uuid4())
            for base_chunk in large_chunk['children_chunks']:
                print("base_chunk", base_chunk)
                data = {
                    'id': str(uuid.uuid4()),
                    'parent_id': parent_id,
                    'vector': get_embedding(base_chunk),
                    "room_uuid": room_uuid,
                    "base_chunk": base_chunk,
                    "parent_chunk": large_chunk['parent_chunk']
                }
                response = opensearch_client.index(
                    index=OPENSEARCH_INDEX_NAME,
                    body=data,
                    id=data['id'],
                    refresh=True
                )
                res = milvus_client.insert(
                    collection_name=MILVUS_COLLECTION_NAME,
                    data=data
                )
    except Exception as e:
        logging.error(e)
