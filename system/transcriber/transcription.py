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


def transcribe_audio_from_s3(bucket: str, key: str, start_timestamp=0):
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        audio_data = response['Body'].read()

        audio_segments = opus_to_segments_pyannote(audio_data, start_timestamp)
        result = []
        trances = transcribe_audio_batch([audio_segment[0] for audio_segment in audio_segments])
        for i, trance in enumerate(trances):
            result.append((trance, audio_segments[i][1]))
        return result
    except Exception as e:
        print(traceback.format_exc())
        raise Exception(f"Error fetching file from S3: {str(e)}")


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
                transcriptions = transcribe_audio_from_s3(bucket, file, start_timestamp=timestamp)
                for transcription in transcriptions:
                    print('transcription!', transcription)
                    for chunk in transcription[0]:
                        transcriptions_result.append(
                            (transcription[1] + chunk['timestamp'][0], username, chunk['text']))

        transcriptions_result.sort(key=lambda x: x[0])
        final_transcription = "\n".join(
            [f"[{username}] {text}\n" for
             timestamp, username, text in transcriptions_result])
        print("final_transcription", final_transcription)

        output = {
            'room_uuid': folder,
            'final_transcription': final_transcription
        }

        producer.produce(f'{bucket}:{folder}', output)
        return final_transcription
    except Exception as e:
        print(traceback.format_exc())
        raise Exception(f"Error processing files from S3: {str(e)}")
