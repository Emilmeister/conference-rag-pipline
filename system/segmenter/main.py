from kafka import KafkaConsumer, KafkaProducer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import logging
import os
import json
import re

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка переменных окружения из .env файла
load_dotenv()

# Получение переменных окружения
SEGMENTER_INPUT_TOPIC = os.getenv('SEGMENTER_INPUT_TOPIC')
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
SEGMENTER_KAFKA_GROUP_ID = os.getenv('SEGMENTER_KAFKA_GROUP_ID')
BASE_CHUNK_SIZE = int(os.getenv('BASE_CHUNK_SIZE'))
LARGE_CHUNK_SIZE = int(os.getenv('LARGE_CHUNK_SIZE'))
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL')
SEGMENTS_OUTPUT_TOPIC = os.getenv('SEGMENTS_OUTPUT_TOPIC')
OPENAI_API_URL = os.getenv('OPENAI_API_URL')

model = init_chat_model(DEFAULT_MODEL, model_provider="openai", base_url=OPENAI_API_URL)

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
    """Удаляет теги <think> и их содержимое из текста.

    Args:
        text: Исходный текст, который может содержать теги <think>.

    Returns:
        Текст без тегов <think> и их содержимого.
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)


# Прослушивание топика входящих сообщений
for message in consumer:
    logging.info('Received message: %s', message)
    room_uuid = message.value.get('room_uuid')
    final_transcription = message.value.get('final_transcription')
    logging.info('room_uuid: %s', room_uuid)
    summary = remove_think_tags(model.invoke("Сгенерируй саммари диалога и ничего больше: \n\n" + final_transcription))
    large_chunks = large_text_splitter.split_text(final_transcription) + large_text_splitter.split_text(summary.text())
    chunks = []
    for large_chunk in large_chunks:
        normal_chunks = normal_text_splitter.split_text(large_chunk)
        chunk_relation = {
            "parent_chunk": large_chunk,
            "children_chunks": normal_chunks
        }
        chunks.append(chunk_relation)
    result = {
        "chunks": chunks,
        "room_uuid": room_uuid
    }
    producer.send(SEGMENTS_OUTPUT_TOPIC, key=room_uuid, value=result)
