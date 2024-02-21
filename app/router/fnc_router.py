from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from fastapi.responses import JSONResponse
import speech_recognition as sr
import os
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
router_functions = APIRouter()

@router_functions.post('/speechtotext')
async def speechtotext(uploaded_file: UploadFile = File(...)):
    # Check if the uploaded file is a webm audio file
    if uploaded_file.content_type not in ['audio/webm']:
        raise HTTPException(status_code=400, detail="This API only accepts WEBM audio files.")

    temp_webm_file_path = None
    temp_wav_file_path = None

    try:
        with NamedTemporaryFile(delete=False, suffix='.webm') as temp_webm_file:
            temp_webm_file.write(uploaded_file.file.read())
            temp_webm_file_path = temp_webm_file.name

        # Convert webm to wav using pydub
        sound = AudioSegment.from_file(temp_webm_file_path, format="webm")
        temp_wav_file_path = temp_webm_file_path.replace('.webm', '.wav')
        sound.export(temp_wav_file_path, format="wav")

        # Initialize the recognizer
        r = sr.Recognizer()
        with sr.AudioFile(temp_wav_file_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Make sure to remove the temporary files
        if temp_webm_file_path and os.path.exists(temp_webm_file_path):
            os.remove(temp_webm_file_path)
        if temp_wav_file_path and os.path.exists(temp_wav_file_path):
            os.remove(temp_wav_file_path)
    print(text)
    return {"text": text}