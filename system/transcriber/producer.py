from kafka import KafkaProducer
from dotenv import load_dotenv
import logging
import os
import json

# Загрузка переменных окружения из .env файла
load_dotenv()
TRANSCRIPTION_OUTPUT_TOPIC = os.getenv('TRANSCRIPTION_OUTPUT_TOPIC')
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS')

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda x: json.dumps(x).encode('utf-8'),
    key_serializer=str.encode
)


def produce(key, value):
    producer.send(TRANSCRIPTION_OUTPUT_TOPIC, key=key, value=value)
