# Audio Transcription API

This API provides a service for transcribing audio files stored in AWS S3 using the Whisper model from Hugging Face.

## Setup

### Using Docker

1. Create a `.env` file with your AWS S3 credentials:
    ```sh
    cp .env.example .env
    ```

2. Build the Docker image:
    ```sh
    docker build -t audio_transcription .
    ```

3. Run the Docker container:
    ```sh
    docker run -p 8000:8000 audio_transcription
    ```

### Without Docker

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/audio_transcription.git
    cd audio_transcription
    ```

2. Create a `.env` file with your AWS S3 credentials:
    ```sh
    cp .env.example .env
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Run the FastAPI server:
    ```sh
    uvicorn main\:app --reload
    ```

## API Endpoints

### Transcribe Audio

- **URL**: `/transcribe/`
- **Method**: `POST`
- **Description**: Transcribes the audio files stored in the specified S3 folder.
- **Request**:
    - **Body**:
        ```json
        {
            "bucket": "your-s3-bucket",
            "folder": "your-folder"
        }
        ```
- **Response**:
    - `200 OK`:
        ```json
        {
            "transcription": "[User1]:[Text]\n[User2]:[Text]\n[User1]:[Text]\n[User2]:[Text]"
        }
        ```
    - `500 Internal Server Error`:
        ```json
        {
            "detail": "Error message"
        }
        ```

## Example Request

```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/transcribe/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "bucket": "your-s3-bucket",
    "folder": "your-folder"
  }'
```

### Запуск

1. Установите Docker, если он еще не установлен.

2. Склонируйте репозиторий:
    ```sh
    git clone https://github.com/yourusername/audio_transcription.git
    cd audio_transcription
    ```

3. Создайте файл `.env` и добавьте в него свои учетные данные AWS S3:
    ```sh
    cp .env.example .env
    ```

4. Соберите Docker-образ:
    ```sh
    docker build -t audio_transcription .
    ```

5. Запустите Docker-контейнер:
    ```sh
    docker run -p 8000:8000 audio_transcription
    ```

Теперь ваш API будет доступен по адресу `http://127.0.0.1:8000`. Вы можете отправить POST-запрос на `/transcribe/` с данными для S3 (bucket, folder) для получения транскрипции всех файлов в указанной папке.
