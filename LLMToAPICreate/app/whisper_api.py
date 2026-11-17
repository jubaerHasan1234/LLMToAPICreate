import os
import tempfile
import urllib.request
from pydantic import BaseModel
from fastapi import APIRouter
from transformers import pipeline


router = APIRouter()

pipe=pipeline("automatic-speech-recognition", model="AiModel/whisper")

class TranscribeRequest(BaseModel):
    audio_url: str

@router.post("/transcribe-whisper")
def transcribe(request: TranscribeRequest):

        #Download audio from url to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            urllib.request.urlretrieve(request.audio_url, tmp.name)
            tmp_path=tmp.name

        # Audio --> Model --> Text
        result= pipe(tmp_path, return_timestamps= True)
        # Remove Temp File
        os.remove(tmp_path)

        return {"content":result["text"]}