import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

class ElectiveRecommendationSystem:
    def __init__(self):
        self.model = BayesianNetwork([
            ('Student Background', 'Department'),
            ('Location of University', 'Department'),
            ('Department', 'Type of Elective'),
            ('Department', 'Course Attributes'),
            ('Type of Elective', 'Course Attributes'),
            ('Professor Names', 'Course Attributes'),
            ('Course Attributes', 'Elective Subjects'),
            ('Term Offered', 'Course Attributes'),
            ('Student Preferences', 'Preferred Learning Mode'),
            ('Preferred Learning Mode', 'Elective Subjects'),
        ])
        
        # Define CPTs
        self.cpd_student_background = TabularCPD('Student Background', 5, [[0.2], [0.2], [0.2], [0.2], [0.2]])
        self.cpd_location_university = TabularCPD('Location of University', 2, [[0.5], [0.5]])
        self.cpd_department = TabularCPD('Department', 3, 
                                         [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 
                                          [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], 
                                          [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]], 
                                         evidence=['Student Background', 'Location of University'], 
                                         evidence_card=[5, 2])
        self.cpd_type_elective = TabularCPD('Type of Elective', 4, 
                                            [[0.4, 0.3, 0.3],  # Department: CS
                                             [0.3, 0.4, 0.3],  # EE
                                             [0.2, 0.2, 0.4],  # ME
                                             [0.1, 0.1, 0.0]], # Sums to 1
                                            evidence=['Department'], 
                                            evidence_card=[3])
        self.cpd_course_attributes = TabularCPD('Course Attributes', 4, 
                                                np.random.dirichlet(np.ones(4), size=240).T, 
                                                evidence=['Department', 'Type of Elective', 'Professor Names', 'Term Offered'], 
                                                evidence_card=[3, 4, 10, 2])
        self.cpd_elective_subjects = TabularCPD('Elective Subjects', 12, 
                                                np.random.dirichlet(np.ones(12), size=16).T, 
                                                evidence=['Course Attributes', 'Preferred Learning Mode'], 
                                                evidence_card=[4, 4])
        self.cpd_preferred_learning_mode = TabularCPD('Preferred Learning Mode', 4, 
                                                      np.random.dirichlet(np.ones(4), size=4).T, 
                                                      evidence=['Student Preferences'], 
                                                      evidence_card=[4])
        self.cpd_student_preferences = TabularCPD('Student Preferences', 4, 
                                                  [[0.25], [0.25], [0.25], [0.25]])
        self.cpd_professor_names = TabularCPD('Professor Names', 10, 
                                              [[0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1]])
        self.cpd_term_offered = TabularCPD('Term Offered', 2, [[0.5], [0.5]])

        # Add CPDs to the model
        self.model.add_cpds(self.cpd_student_background, self.cpd_location_university, self.cpd_department, 
                            self.cpd_type_elective, self.cpd_course_attributes, self.cpd_elective_subjects, 
                            self.cpd_preferred_learning_mode, self.cpd_student_preferences, 
                            self.cpd_professor_names, self.cpd_term_offered)

        # Validate the model
        assert self.model.check_model()

        # Inference
        self.inference = VariableElimination(self.model)

    def recommend_electives(self, user_input):
        evidence = {
            'Student Background': int(user_input['Student Background']),
            'Location of University': int(user_input['Location of University']),
            'Department': int(user_input['Department']),
            'Type of Elective': int(user_input['Type of Elective']),
            'Professor Names': int(user_input['Professor Names']),
            'Term Offered': int(user_input['Term Offered']),
            'Student Preferences': int(user_input['Student Preferences']),
            'Preferred Learning Mode': int(user_input['Preferred Learning Mode']),
        }
        
        # Query the network to get the probabilities of elective subjects
        query_result = self.inference.query(variables=['Elective Subjects'], evidence=evidence)
        elective_probabilities = query_result.values
        
        # Get the top 5 electives with their probabilities
        elective_names = [
            "Advanced Control", "Bayesian Inference and Gaussian Processes",
            "Cognitive Robotics", "Deep Learning for Robot Vision",
            "Deep Learning Foundations", "Entrepreneurship in Robotics and Computer Science",
            "Human-centered Interaction in Robotics", "Natural Language Processing",
            "Probabilistic Reasoning", "Robot Manipulation",
            "Scientific Experimentation and Evaluation", "Advanced Scientific Working"
        ]
        
        top_5_indices = np.argsort(elective_probabilities)[-5:][::-1]
        top_5_electives = [(elective_names[i], elective_probabilities[i]) for i in top_5_indices]
        
        return top_5_electives
