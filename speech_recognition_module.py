import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import os

class SpeechRecognition:
    def __init__(self):
        self.recogniser = sr.Recognizer()

    def listen(self):
        audio = self._listen_to_speech()
        text = self._speech_to_text(audio)
        return text

    def _listen_to_speech(self):
        with sr.Microphone() as source:
            print("Using system default Microphone...")
            self.recogniser.adjust_for_ambient_noise(source)
            audio = self.recogniser.listen(source)
            print("Listening...")
            return audio

    def _speech_to_text(self, audio):
        text = ""
        try:
            text = self.recogniser.recognize_google(audio)
        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")
        except sr.RequestError as e:
            print(f"Request to Google Web Speech API failed; {e}")
        return text

    def speak(self, text):
        tts = gTTS(text=text, lang='en')
        tts.save("response.mp3")
        playsound("response.mp3")
        os.remove("response.mp3")
