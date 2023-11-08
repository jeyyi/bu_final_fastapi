from fastapi import APIRouter
import speech_recognition as sr
import os
router_functions = APIRouter()

@router_functions.post('/speechtotext')
def speechtotext():
    # initialize the recognizer
    filename = 'app/resources/sample.wav'
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        # listen for the data (load audio to memory)
        audio_data = r.record(source)
        # recognize (convert from speech to text)
        text = r.recognize_google(audio_data)
        return text