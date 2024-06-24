import os
import cv2
import face_recognition
from pathlib import Path
from qibullet import SimulationManager
import gtts
from playsound import playsound
import threading
import time

class PepperAssistant:
    def __init__(self, known_faces_dir):
        # Initialize the simulation manager
        self.simulation_manager = SimulationManager()
        self.client = self.simulation_manager.launchSimulation(gui=True)
        self.pepper = self.simulation_manager.spawnPepper(self.client, spawn_ground_plane=True)
        
        # Set Pepper's initial posture
        self.pepper.goToPosture("Stand", 0.6)
        self.pepper.setAngles("HeadYaw", 0.0, 0.6)

        # Initialize face recognition
        self.known_faces = self._load_known_faces(known_faces_dir)
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            raise Exception("Could not open video device")
        self.frame = None
        self.detected_location = None
        self.detected_person = None
        self.last_greeted_person = None
        self.stop_flag = False
        self.polling_rate = 0.008  # In seconds
        self.non_recognised_frames_limit = 3
        self.non_recognised_frames = 0
        self.greeted = False

    #Loading known faces
    def _load_known_faces(self, known_faces_dir):
        '''
        This method loads face images from the specified directory. 
        For each .jpg file, it extracts the person's name from the filename,
        loads the face encoding, and stores it in a dictionary. 
        If an error occurs, it is printed out.
        '''
        result = {}
        for file in os.listdir(known_faces_dir):
            if file.endswith(".jpg"):
                person = Path(file).stem
                try:
                    result[person] = self._memorise_face(os.path.join(known_faces_dir, file))
                    print(f"Loaded face for {person}")
                except Exception as e:
                    print(f"Error loading face for {person}: {e}")
        if not result:
            print("No faces to load")
        return result

    @staticmethod
    def _memorise_face(image_path):
        '''
        This static method loads an image file, 
        finds face encodings, and returns the first encoding found. 
        If no face is found, it returns None
        '''
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            print(f"Could not find a face in {image_path}")
            return None
        encoding = encodings[0]
        return encoding
    
    # Face detection and recognition

    def _rect_area(self, top, right, bottom, left):
        '''
        Calculates the area of a rectangle i.e face bounding box
        '''
        area = abs(top - bottom) * abs(left - right)
        return area

    def _draw_box(self):
        '''
        If a face is detected, this method draws a red rectangle 
        around it on the current video frame.
        '''
        if self.detected_location:
            (top, right, bottom, left) = self.detected_location
            # Draw a rectangle around the face
            cv2.rectangle(self.frame, (left, top), (right, bottom), (0, 0, 255), 2)

    def _label_box(self):
        '''
        This method labels the detected face with the person's name above the bounding box.
        '''
        if self.detected_location:
            x = self.detected_location[3]
            y = self.detected_location[2]
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(self.frame, self.detected_person, (x + 6, y - 6), font, 0.5, (255, 255, 255), 1)
            
    
    #face detection loop
    def _detect_face(self):
        '''
        This method continuously captures frames from the video device. 
        It detects face locations, selects the largest (dominant) face, 
        and increments a counter if no face is found.
        If a face is found or a limit is reached,
        it sets the detected location and calls the recognition method.
        '''
        while not self.stop_flag:
            ret, self.frame = self.video_capture.read()
            if ret:
                detected_locations = face_recognition.face_locations(self.frame)
                print(f"Detected face locations: {detected_locations}")  # Debug: Print detected face locations
                dominant_location = None
                prev_bounding_area = 0

                for (top, right, bottom, left) in detected_locations:
                    bounding_area = self._rect_area(top, right, bottom, left)
                    if bounding_area > prev_bounding_area:
                        dominant_location = (top, right, bottom, left)
                        prev_bounding_area = bounding_area

                if dominant_location is None:
                    self.non_recognised_frames += 1

                if self.non_recognised_frames == self.non_recognised_frames_limit or dominant_location:
                    self.detected_location = dominant_location
                    self.non_recognised_frames = 0

                if dominant_location:
                    self._recognise_face()
                else:
                    self.detected_person = None

    def _recognise_face(self):
        '''
        This method compares the detected face encoding with known faces.
        If a match is found, it sets the detected person's name.
        '''
        detected_location = self.detected_location
        if detected_location:
            detected_encoding = face_recognition.face_encodings(self.frame, [detected_location])
            if detected_encoding:
                detected_encoding = detected_encoding[0]
                self.detected_person = "Unknown"
                for person, person_encoding in self.known_faces.items():
                    if person_encoding is not None:
                        is_known = face_recognition.compare_faces([person_encoding], detected_encoding)[0]
                        face_distance = face_recognition.face_distance([person_encoding], detected_encoding)[0]
                        print(f"Comparing with {person}: {is_known}, distance: {face_distance}")
                        if is_known:
                            self.detected_person = person
                            break
                print(f"Recognized person: {self.detected_person}")  # Debug: Print recognized person
            else:
                print("No encoding found for the detected face")
        else:
            print("No detected location")
            
    #interaction and greeting

    def _speak(self, text):
        '''
        Google's Text-to-Speech to generate an audio file from the text and plays it.
        '''
        tts = gtts.gTTS(text)
        tts.save("output.mp3")
        playsound("output.mp3")
        os.remove("output.mp3")

    def greet_person(self):
        '''
        Pepper robot's greeting behavior. 
        If a recognized person is detected, it greets them by name and performs gestures.
        If no specific person is recognized, it performs a standard greeting.
        '''
        if self.detected_person and self.detected_person != "Unknown":
            if self.detected_person != self.last_greeted_person:
                self.pepper.goToPosture("Stand", 0.6)
                self.pepper.setAngles("HeadYaw", 0.0, 0.6)
                self.pepper.setAngles("HeadPitch", -0.3, 0.6)  # Nodding gesture
                time.sleep(1)
                self.pepper.setAngles("HeadPitch", 0.0, 0.6)
                
                # Simulate talking gestures
                self.pepper.setAngles("LShoulderPitch", 0.5, 0.6)
                self.pepper.setAngles("RShoulderPitch", 0.5, 0.6)
                self.pepper.setAngles("LElbowYaw", -1.0, 0.6)
                self.pepper.setAngles("RElbowYaw", 1.0, 0.6)
                
                # Speak while gesturing
                greeting = f"Hello {self.detected_person}, how can I assist you in finding a suitable elective course?"
                print(greeting)  # Debug: Print the greeting
                self._speak(greeting)
                self.last_greeted_person = self.detected_person
                
                # Reset arms
                self.pepper.setAngles("LShoulderPitch", 1.5, 0.6)
                self.pepper.setAngles("RShoulderPitch", 1.5, 0.6)
                self.pepper.setAngles("LElbowYaw", -0.5, 0.6)
                self.pepper.setAngles("RElbowYaw", 0.5, 0.6)
        else:
            if self.last_greeted_person != "standard":
                # Standard greeting
                self.pepper.goToPosture("Stand", 0.6)
                self.pepper.setAngles("HeadYaw", 0.0, 0.6)
                standard_greeting = "Hello, how can I assist you today?"
                print(standard_greeting)  # Debug: Print the greeting
                self._speak(standard_greeting)
                self.last_greeted_person = "standard"
                
                
    #main loop RUnning the assistant            

    def run(self):
        '''
        This method starts a separate thread for face detection, 
        then enters a loop to update the video feed and greet detected people. 
        It stops when 'q' is pressed, releasing resources and stopping the simulation.
        '''
        face_thread = threading.Thread(target=self._detect_face, args=(), daemon=True)
        face_thread.start()
        try:
            while not self.stop_flag:
                if self.frame is not None and self.frame.size > 0:
                    self._draw_box()
                    self._label_box()
                    self.greet_person()
                    cv2.imshow('Video Feed', self.frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_flag = True
        except Exception as e:
            print(f"Error: {e}")
        finally:
            face_thread.join()
            self.video_capture.release()
            cv2.destroyAllWindows()
            self.simulation_manager.stopSimulation(self.client)

if __name__ == "__main__":
    known_faces_dir = "known_faces/"
    pepper_assistant = PepperAssistant(known_faces_dir)
    pepper_assistant.run()




