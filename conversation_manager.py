import spacy
from spacy.matcher import Matcher
from speech_recognition_module import SpeechRecognition
import time
class ConversationManager:
    def __init__(self):
        self.sr = SpeechRecognition()
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        
        # Define patterns
        self.elective_pattern = [{"POS": "PROPN"}, {"POS": "PROPN"}]
        self.title_patterns = [
            [{"LOWER": {"IN": ["prof", "professor"]}}, {"IS_ALPHA": True}],
            [{"LOWER": {"IN": ["dr", "doctor"]}}, {"IS_ALPHA": True}],
            [{"LOWER": "m.sc"}, {"IS_ALPHA": True}]
        ]

        self.matcher.add("ELECTIVE_OPTION", [self.elective_pattern])
        for pattern in self.title_patterns:
            self.matcher.add("TITLE", [pattern])

    def extract_information(self, text, label_type=None, pattern_name=None):
        doc = self.nlp(text)
        entities = []
        if label_type:
            entities = [ent.text for ent in doc.ents if ent.label_ == label_type]
        if pattern_name:
            matches = self.matcher(doc)
            entities += [doc[start:end].text for match_id, start, end in matches if self.nlp.vocab.strings[match_id] == pattern_name]
        return entities

    def get_user_input(self, prompt):
        self.sr.speak(prompt)
        print(prompt)
        return self.sr.listen()

    def main(self):
        # Initialize preferences dictionary
        preferences = {
            "elective_options": [],
            "professor_names": [],
            "learning_mode": "",
            "max_participants": "",
            "term_offered": "",
            "student_background": "",
            "course_attributes": []
        }

        # Collect preferences through dialogue
        elective_options = self.get_user_input("What is your preferred elective options:")
        # print(elective_options)
        # elective_options = "I want to study Data Science, Machine Learning and Autonomous Systems"
        # elective_options = "Autonomous Systems is my favorite subject"
        preferences["elective_options"] = self.extract_information(elective_options, pattern_name="ELECTIVE_OPTION")

        professor_names = self.get_user_input("Who is your preferred professors")
        # professor_names = "Professor Arno is my favorite"
        preferences["professor_names"] = self.extract_information(professor_names, pattern_name="TITLE")

        learning_mode = self.get_user_input("What is your preferred learning mode online, offline, hybrid")
        preferences["learning_mode"] = learning_mode.lower()

        max_participants = self.get_user_input("What is the maximum number of participants you prefer?")
        preferences["max_participants"] = self.extract_information(max_participants, label_type="CARDINAL")

        term_offered = self.get_user_input("Which term do you prefer?")
        preferences["term_offered"] = self.extract_information(term_offered, label_type="DATE")

        student_background = self.get_user_input("What is your student background undergraduate, graduate, postgraduate")
        preferences["student_background"] = student_background.lower()

        course_attributes = self.get_user_input("What course attributes do you prefer core, elective, optional")
        # course_attributes =  " I prefer optional"
        preferences["course_attributes"] = [attr.strip().lower() for attr in course_attributes.split(",")]

        # Display the collected preferences
        collected_preferences = "\nCollected Preferences:\n" + "\n".join([f"{key.capitalize()}: {value}" for key, value in preferences.items()])
        # self.sr.speak(collected_preferences)
        print(collected_preferences)

if __name__ == "__main__":
    cm = ConversationManager()
    cm.main()
