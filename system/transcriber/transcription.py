import torch
from transformers import pipeline
import librosa
import io
import boto3
from dotenv import load_dotenv
import os
import re
import producer
import traceback
from segmenter import opus_to_segments_pyannote
from datetime import datetime
from psycopg2 import sql
import psycopg2

load_dotenv()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    "automatic-speech-recognition",
    model=os.getenv('WHISPER_MODEL_NAME', 'openai/whisper-large-v3'),
    chunk_length_s=30, stride_length_s=(10, 5),
    device=device,
)

s3 = boto3.client(
    's3',
    endpoint_url=os.getenv('S3_URL'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

def transcribe_audio(audio_data: bytes):
    audio_input, sample_rate = librosa.load(io.BytesIO(audio_data), sr=16000)
    prediction = pipe(
        audio_input,
        batch_size=int(os.getenv('WHISPER_PARALLEL', '3')),
        return_timestamps=True,
        generate_kwargs={
            'language': 'ru',
            'num_beams': int(os.getenv('NUM_BEAMS', '10')),
            'temperature': float(os.getenv('WHISPER_TEMPERATURE', '0.1')),
        }
    )["chunks"]
    return prediction


def transcribe_audio_batch(list_of_audio_files: list):
    inputs = [librosa.load(io.BytesIO(audio_data), sr=16000)[0] for audio_data in list_of_audio_files]
    predictions = pipe(
        inputs,
        batch_size=int(os.getenv('WHISPER_PARALLEL', '3')),
        return_timestamps=True,
        generate_kwargs={
            'language': 'ru',
            'num_beams': int(os.getenv('NUM_BEAMS', '10')),
            'temperature': float(os.getenv('WHISPER_TEMPERATURE', '0.1')),
        }
    )
    return [prediction["chunks"] for prediction in predictions]


def transcribe_audio_from_s3(room_uuid: str, bucket: str, key: str, username: str, start_timestamp=0):
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        audio_data = response['Body'].read()

        audio_segments = opus_to_segments_pyannote(audio_data, start_timestamp)
        result = []
        trances = transcribe_audio_batch([audio_segment[0] for audio_segment in audio_segments])

        save_to_s3_and_db(audio_segments, username, room_uuid, trances)

        for i, trance in enumerate(trances):
            result.append((trance, audio_segments[i][1]))
        return result
    except Exception as e:
        print(traceback.format_exc())
        raise Exception(f"Error fetching file from S3: {str(e)}")



def save_to_s3_and_db(audio_segments, user_login, room_uuid, trances):
    '''
    :param audio_segments: Список аудио сегментов
    :param user_login: Логин пользователя
    :param room_uuid: UUID комнаты
    :param trances: Список транскрибированных текстов
    :return:
    '''
    # Подключение к базе данных
    conn = psycopg2.connect(
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT')
    )
    cursor = conn.cursor()

    for i, audio_segment in enumerate(audio_segments):
        predicted_text = trances[i][0]['text']
        (data, segment_start_timestamp, segment_end_timestamp) = audio_segment
        s3_audiofragment_path = f'{room_uuid}/{user_login}_{segment_start_timestamp}_{segment_end_timestamp}.wav'
        s3_bucket = os.getenv('S3_AUDIO_CHUNKS_BUCKET')

        # Сохранение аудио в S3
        s3.put_object(Bucket=s3_bucket, Key=s3_audiofragment_path, Body=data)

        # Вставка данных в базу данных
        insert_query = sql.SQL("""
            INSERT INTO transcription.transcription (
                room_uuid, timestamp_begin, timestamp_end, predicted_text, user_login, s3_audiofragment_path, s3_bucket
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """)
        cursor.execute(insert_query, (
            room_uuid,
            datetime.fromtimestamp(segment_start_timestamp / 1000),
            datetime.fromtimestamp(segment_end_timestamp / 1000),
            predicted_text,
            user_login,
            s3_audiofragment_path,
            s3_bucket
        ))

    # Фиксация изменений и закрытие соединения
    conn.commit()
    cursor.close()
    conn.close()



def transcribe_audio_from_s3_folder(bucket: str, folder: str):
    s3 = boto3.client(
        's3',
        endpoint_url=os.getenv('S3_URL'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=folder)
        files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].startswith(f"{folder}/room")]

        transcriptions_result = []
        for file in files:
            print(file)
            match = re.match(r'room\.(.+?)\.(.+?)\.(\d+)\.ogg', file.split("/")[1])
            if match:
                uuid, username, timestamp = match.groups()
                timestamp = int(timestamp)
                transcriptions = transcribe_audio_from_s3(uuid, bucket, file, username, start_timestamp=timestamp)
                for transcription in transcriptions:
                    print('transcription!', transcription)
                    for chunk in transcription[0]:
                        transcriptions_result.append(
                            (transcription[1] + chunk['timestamp'][0], username, chunk['text']))

        transcriptions_result.sort(key=lambda x: x[0])
        final_transcription = "\n".join(
            [f"[{datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')}, {username}] {text}" for
             timestamp, username, text in transcriptions_result])
        print("final_transcription", final_transcription)

        llm_input = {
            'systemPrompt': '''
                Пиши ответы на русском языке!
                Ты — эксперт в суммаризации текстов.
                Твоя задача — создать краткое и точное резюме следующего текста, сохраняя все ключевые моменты и основные идеи.
                Текст должен быть сокращен до 100-300 слов, но при этом оставаться информативным и понятным. 
                Пожалуйста, избегай использования излишне сложных или специализированных терминов, если это не требуется для понимания сути.
                Текст для суммаризации заключен в тройные скобки. Не пиши ничего лишнего в результат, кроме самого текста резюме.
            ''',
            'queryPrompt': f'(((%s)))',
            'args': [final_transcription],
            'choices': 2,
            'temperature': 0.1,
            'kafkaOutputTopic': 'OLLAMA_LLM_OUTPUT',
            'kafkaOutputKey': folder,
        }

        producer.produce(f'{bucket}:{folder}', llm_input)
        return final_transcription
    except Exception as e:
        print(traceback.format_exc())
        raise Exception(f"Error processing files from S3: {str(e)}")
