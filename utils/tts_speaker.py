# utils/tts_speaker.py

from gtts import gTTS
import os
import pygame
import time
import random

pygame.mixer.init()

def speak(text):
    filename = f"temp_audio_{random.randint(1000,9999)}.mp3"
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    os.remove(filename)
