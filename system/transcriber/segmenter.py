import dotenv
from pyannote.audio import Pipeline
from pydub import AudioSegment
from dotenv import load_dotenv
import io
import os
import uuid
import wave

load_dotenv()


def export_segments(audio, segments, start_timestamp=0, output_dir="output"):
    """Нарезка аудио по временным меткам и сохранение сегментов"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio_segments = []

    pause_millis = int(os.getenv('SEGMENTER_PAUSE_MILLIS', '1000'))

    for i, (start_sec, end_sec) in enumerate(segments):
        # Конвертация секунд в миллисекунды
        start_ms = int(max(start_sec * 1000 - pause_millis, 0))
        end_ms = int(min(end_sec * 1000, end_sec * 1000 + pause_millis))

        # Вырезаем сегмент
        segment = audio[start_ms:end_ms]

        segment_start_timestamp = start_timestamp + start_ms
        segment_end_timestamp = start_timestamp + end_ms

        buffer = io.BytesIO()
        segment.export(buffer, format="wav")

        audio_segments.append((buffer.read(), segment_start_timestamp, segment_end_timestamp))
    return audio_segments


def opus_to_segments_pyannote(audio_data, start_timestamp=0):
    temp_audio_file = f"{uuid.uuid4()}.ogg"
    with open(temp_audio_file, 'wb') as f:
        f.write(audio_data)
        f.close()

    # Загружаем оригинальное аудио для нарезки
    original_audio = AudioSegment.from_file(temp_audio_file, codec="opus")

    # Удаляем временный файл
    if os.path.exists(temp_audio_file):
        os.remove(temp_audio_file)

    # Конвертируем в WAV для Pyannote (требование модели)
    temp_wav = f"{uuid.uuid4()}.wav"
    original_audio.export(temp_wav, format="wav")

    # Обработка через Pyannote
    pipeline = Pipeline.from_pretrained(
        "pyannote/voice-activity-detection",
        use_auth_token='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'
    )
    output = pipeline(temp_wav)

    # Удаляем временный файл
    if os.path.exists(temp_wav):
        os.remove(temp_wav)

    # Извлекаем таймштампы
    segments = [(segment.start, segment.end) for segment in output.get_timeline().support()]

    # Нарезаем аудио на сегменты
    audio_segments = export_segments(original_audio, segments, start_timestamp=start_timestamp)

    return audio_segments