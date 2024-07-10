import spacy
from spacy.matcher import Matcher
from speech_recognition_module import SpeechRecognition
import numpy as np
from difflib import SequenceMatcher

class ConversationManager:
    def __init__(self):
        # Initialize the SpeechRecognition class
        self.sr = SpeechRecognition()
        # Load the spaCy language model
        self.nlp = spacy.load("en_core_web_md")  # Using a larger model for better embeddings

        # Options list for internal matching
        self.options = {
            'Student Background': ["Computer Science", "Electrical Engineering", "Mechanical Engineering", "Mechatronics", "Autonomous Systems"],
            'Location of University': ["Campus Sankt Augustin", "Campus Rheinbach"],
            'Department': ["Computer Science", "Electrical Engineering", "Mechanical Engineering"],
            'Type of Elective': ["Core", "Specialization", "Research", "Interdisciplinary"],
            'Professor Names': ["Prof. Dr. Wolfgang Borutzky", "Prof. Dr. Alexander Asteroth", "Dr. Aleksandar Mitrevski", "Prof. Dr. Sebastian Houben", "Prof. Dr. Joern Hees", "Prof. Dr. Erwin Prassler", "Prof. Dr. Teena Hassan", "M.Sc. Tim Metzler", "M.Sc. Youssef Mahmoud Youssef", "Prof. Dr. Paul G. Plöger"],
            'Term Offered': ["Summer Semester", "Winter Semester"],
            'Student Preferences': ["AI", "Robotics", "Data Science", "Sustainable Energy"],
            'Preferred Learning Mode': ["Lecture", "Lab", "Seminar", "Workshop"]
        }


    def get_user_input(self, prompt, options):
        # Prompt the user and get their input using speech recognition
        self.sr.speak(prompt)
        print(prompt)
        print("Options:", options)  # Display options on screen
        response = self.sr.listen(timeout=10)  # Increase timeout to 10 seconds
        print("User response:", response)
        return response

    def match_keyword_to_option(self, keyword, options):
        if not keyword:
            return None
        keyword_doc = self.nlp(keyword.lower())
        
        # Calculate token similarity
        similarities = [keyword_doc.similarity(self.nlp(option.lower())) for option in options]
        
        # Calculate string similarity using Levenshtein distance
        string_similarities = [SequenceMatcher(None, keyword.lower(), option.lower()).ratio() for option in options]
        
        # Combine the similarities
        combined_similarities = [0.5 * sim + 0.5 * str_sim for sim, str_sim in zip(similarities, string_similarities)]
        
        best_match_idx = int(np.argmax(combined_similarities))
        return best_match_idx

    def collect_preferences(self):
        # Initialize preferences dictionary to store user preferences
        preferences = {
            "Student Background": "",
            "Location of University": "",
            "Department": "",
            "Type of Elective": "",
            "Professor Names": "",
            "Term Offered": "",
            "Student Preferences": "",
            "Preferred Learning Mode": ""
        }

        for key in preferences.keys():
            while True:
                prompt = f"Please state your preference for {key.replace('_', ' ')}:"
                response = self.get_user_input(prompt, self.options[key])
                matched_idx = self.match_keyword_to_option(response, self.options[key])
                if matched_idx is not None:
                    preferences[key] = matched_idx
                    print(f"{key}: {response} (matched to {self.options[key][matched_idx]})")
                    break
                else:
                    print(f"Could not understand your response for {key}. Please try again.")
                    self.sr.speak(f"Could not understand your response for {key}. Please try again.")

        return preferences