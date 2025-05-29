from transcription import transcribe_audio_from_s3_folder
import os
import logging
from kafka import KafkaConsumer, KafkaProducer
from dotenv import load_dotenv
import json

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка переменных окружения из .env файла
load_dotenv()

# Получение переменных окружения
TRANSCRIPTION_INPUT_TOPIC = os.getenv('TRANSCRIPTION_INPUT_TOPIC')
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS')
TRANSCRIBER_KAFKA_GROUP_ID = os.getenv('TRANSCRIBER_KAFKA_GROUP_ID')




# Инициализация Kafka Consumer и Producer
consumer = KafkaConsumer(
    TRANSCRIPTION_INPUT_TOPIC,
    group_id=TRANSCRIBER_KAFKA_GROUP_ID,
    bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
    session_timeout_ms=30*60*1000
)

logging.info('Partitions of the topic: %s', consumer.partitions_for_topic(TRANSCRIPTION_INPUT_TOPIC))

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda x: json.dumps(x).encode('utf-8'),
)

# Прослушивание топика входящих сообщений
for message in consumer:
    logging.info('Received message: %s', message)
    folder = message.value.get('folder')
    bucket = message.value.get('bucket')
    logging.info('folder, bucket: %s, %s', folder, bucket)
    transcribe_audio_from_s3_folder(bucket, folder)
    logging.info('processing_end: %s, %s', folder, bucket)


