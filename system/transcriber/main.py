from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transcription import transcribe_audio_from_s3_folder
import uvicorn

app = FastAPI()


class S3Folder(BaseModel):
    bucket: str
    folder: str


@app.post("/transcribe/")
async def transcribe(s3_folder: S3Folder):
    try:
        transcription = transcribe_audio_from_s3_folder(s3_folder.bucket, s3_folder.folder)
        return {"transcription": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
