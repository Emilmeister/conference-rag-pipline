from agents import set_trace_processors, set_default_openai_client, set_default_openai_api, Agent, trace, gen_trace_id, \
    Runner
from agents.models import openai_provider
from kafka import KafkaConsumer, KafkaProducer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import logging
import os
import json
import re
import ssl

from openai import AsyncOpenAI
from openai.types.responses import EasyInputMessageParam
from phoenix.otel import register
from pydantic import BaseModel

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка переменных окружения из .env файла
load_dotenv()

ssl._create_default_https_context = ssl._create_unverified_context

PHOENIX_TRACE_URL = os.getenv("PHOENIX_TRACE_URL")
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")
RETRIEVER_URL = os.getenv("RETRIEVER_URL")

# configure the Phoenix tracer
set_trace_processors([])
tracer_provider = register(
    project_name=PHOENIX_PROJECT_NAME,
    endpoint=PHOENIX_TRACE_URL,
    auto_instrument=True
)

set_default_openai_client(AsyncOpenAI(base_url=OPENAI_API_URL, api_key=OPENAI_API_KEY, timeout=60 * 5))
set_default_openai_api('chat_completions')
openai_provider.DEFAULT_MODEL = DEFAULT_MODEL

SEGMENTER_INPUT_TOPIC = os.getenv('SEGMENTER_INPUT_TOPIC')
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
SEGMENTER_KAFKA_GROUP_ID = os.getenv('SEGMENTER_KAFKA_GROUP_ID')
BASE_CHUNK_SIZE = int(os.getenv('BASE_CHUNK_SIZE'))
LARGE_CHUNK_SIZE = int(os.getenv('LARGE_CHUNK_SIZE'))
SEGMENTS_OUTPUT_TOPIC = os.getenv('SEGMENTS_OUTPUT_TOPIC')
OPENAI_API_URL = os.getenv('OPENAI_API_URL')


class SummarizerOutput(BaseModel):
    reasoning: str
    summary: str
    dialog_title: str


agent = Agent(
    name="Summarizer",
    instructions="""Тебе на вход подадут транскрибацию диалога из системы видоеконференций. 
    Сгенерируй краткое содержание диалога и возможное его название в системе и ничего больше.""",
    output_type=SummarizerOutput
)

large_text_splitter = RecursiveCharacterTextSplitter(chunk_size=LARGE_CHUNK_SIZE, chunk_overlap=0)
normal_text_splitter = RecursiveCharacterTextSplitter(chunk_size=BASE_CHUNK_SIZE,
                                                      chunk_overlap=int(BASE_CHUNK_SIZE / 4))

# Инициализация Kafka Consumer и Producer
consumer = KafkaConsumer(
    SEGMENTER_INPUT_TOPIC,
    group_id=SEGMENTER_KAFKA_GROUP_ID,
    bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda x: json.dumps(x).encode('utf-8'),
    key_serializer=str.encode
)

logging.info('Partitions of the topic: %s', consumer.partitions_for_topic(SEGMENTER_INPUT_TOPIC))


def remove_think_tags(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)


# Прослушивание топика входящих сообщений
for message in consumer:
    logging.info('Received message: %s', message)
    room_uuid = message.value.get('room_uuid')
    final_transcription = message.value.get('final_transcription')
    logging.info('room_uuid: %s', room_uuid)
    trace_id = gen_trace_id()
    with trace(workflow_name="Question Answering", trace_id=trace_id):
        result = Runner.run_sync(
            starting_agent=agent,
            input=[EasyInputMessageParam(role='user', content=final_transcription)]
        )

    summary_result = SummarizerOutput.model_validate(result.final_output)

    large_chunks = (large_text_splitter.split_text(final_transcription)
                    + large_text_splitter.split_text(remove_think_tags(summary_result.summary)))
    chunks = []
    for large_chunk in large_chunks:
        normal_chunks = normal_text_splitter.split_text(large_chunk)
        chunk_relation = {
            "parent_chunk": large_chunk,
            "children_chunks": normal_chunks,
            "meeting_label": summary_result.dialog_title
        }
        chunks.append(chunk_relation)
    result = {
        "chunks": chunks,
        "room_uuid": room_uuid
    }
    producer.send(SEGMENTS_OUTPUT_TOPIC, key=room_uuid, value=result)
