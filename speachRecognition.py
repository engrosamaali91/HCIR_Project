import os
import sys
import time
from threading import Thread
import speech_recognition
from gtts import gTTS
import playsound

class SpeechRecognition:
    def __init__(self):
        # Initialize the recognizer from the speech_recognition library
        self.recogniser = speech_recognition.Recognizer()

    def listen(self):
        # Capture audio from the microphone
        audio = self._listen_to_speech()
        # Convert the captured audio to text
        text = self._speech_to_text(audio)
        if text:
            # If text was successfully recognized, convert it to speech
            self._text_to_speech(text)
        return text

    def _listen_to_speech(self):
        # Use the microphone as the audio source
        with speech_recognition.Microphone() as source:
            print("Using system default Microphone...")
            # Adjust the recognizer sensitivity to ambient noise
            self.recogniser.adjust_for_ambient_noise(source)
            print("Listening...")
            # Listen for the first phrase and extract it into audio data
            audio = self.recogniser.listen(source)
            return audio

    def _speech_to_text(self, audio):
        # Initialize an empty string for the recognized text
        text = ""
        try:
            # Use Google's speech recognition to convert audio to text
            text = self.recogniser.recognize_google(audio)
            print(f"Recognized text: {text}")
        except speech_recognition.UnknownValueError:
            # If the speech was unintelligible
            print("Sorry, could not understand audio.")
        except speech_recognition.RequestError as e:
            # If there was an error with the request to Google's API
            print(f"Request to Google Web Speech API failed; {e}")
        return text

    def _text_to_speech(self, text):
        print("Converting text to speech...")
        # Convert the text to speech using gTTS
        tts = gTTS(text=text, lang='en')
        # Save the speech as an MP3 file
        tts.save("output.mp3")
        # Play the MP3 file
        playsound.playsound("output.mp3")
        # Remove the MP3 file after playing
        os.remove("output.mp3")

if __name__ == "__main__":
    # Create an instance of the SpeechRecognition class
    sr = SpeechRecognition()
    # Continuously listen and respond to speech
    while True:
        # Listen for speech and convert it to text and then to speech
        text = sr.listen()
        # Print the recognized text
        print(text)
