import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import os

class SpeechRecognition:
    def __init__(self):
        # Initialize the recognizer from the speech_recognition library
        self.recogniser = sr.Recognizer()

    def listen(self, timeout=10):
        # Capture audio from the microphone
        audio = self._listen_to_speech(timeout)
        # Convert the captured audio to text
        text = self._speech_to_text(audio)
        return text

    def _listen_to_speech(self, timeout):
        # Use the microphone as the audio source
        with sr.Microphone() as source:
            print("Using system default Microphone...")
            # Adjust the recognizer sensitivity to ambient noise
            self.recogniser.adjust_for_ambient_noise(source)
            # Listen for the first phrase and extract it into audio data
            print("Listening...")
            audio = self.recogniser.listen(source, timeout=timeout)
            return audio

    def _speech_to_text(self, audio):
        # Initialize an empty string for the recognized text
        text = ""
        # Use Google's speech recognition to convert audio to text
        try:
            text = self.recogniser.recognize_google(audio)
            print("Recognized text:", text)
        except sr.UnknownValueError:
            # If the speech was unintelligible
            print("Sorry, could not understand audio.")
        except sr.RequestError as e:
            # If there was an error with the request to Google's API
            print(f"Request to Google Web Speech API failed; {e}")
        return text

    def speak(self, text):
         # Convert the text to speech using gTTS
        tts = gTTS(text=text, lang='en')
        # Save the speech as an MP3 file
        tts.save("response.mp3")
        # Play the MP3 file
        playsound("response.mp3")
        # Remove the MP3 file after playing to clean up
        os.remove("response.mp3")
