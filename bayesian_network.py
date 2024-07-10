import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

class ElectiveRecommendationSystem:
    def __init__(self):
        self.model = BayesianNetwork([
            ('Student Background', 'Department'),
            ('Student Background', 'Difficulty Level'),
            ('Location of University', 'Department'),
            ('Department', 'Type of Elective'),
            ('Department', 'Professor Names'),
            ('Type of Elective', 'Difficulty Level'),
            ('Difficulty Level', 'Elective Subjects'),
            ('Professor Names', 'Elective Subjects'),
            ('Preferred Branch', 'Preferred Learning Mode'),
            ('Preferred Learning Mode', 'Elective Subjects'),
            ('Term Offered', 'Difficulty Level')
        ])

        # Define CPTs
        self.cpd_student_background = TabularCPD(variable='Student Background', variable_card=5, values=[[0.2], [0.2], [0.2], [0.2], [0.2]])
        self.cpd_location_university = TabularCPD(variable='Location of University', variable_card=2, values=[[0.5], [0.5]])
        self.cpd_department = TabularCPD(variable='Department', variable_card=3,
                                         values=[[0.7, 0.5, 0.6, 0.7, 0.5, 0.4, 0.6, 0.7, 0.5, 0.4],  
                                                 [0.2, 0.3, 0.3, 0.2, 0.3, 0.4, 0.3, 0.2, 0.3, 0.4],
                                                 [0.1, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2]], 
                                         evidence=['Student Background', 'Location of University'], 
                                         evidence_card=[5, 2])
        self.cpd_professor_names = TabularCPD(variable='Professor Names', variable_card=6, 
                                      values=[
                                          [0.16, 0.15, 0.09],
                                          [0.17, 0.15, 0.10],
                                          [0.17, 0.20, 0.11],
                                          [0.17, 0.15, 0.20],
                                          [0.16, 0.20, 0.25],
                                          [0.17, 0.15, 0.25]
                                      ], 
                                      evidence=['Department'], 
                                      evidence_card=[3])
        self.cpd_type_elective = TabularCPD(variable='Type of Elective', variable_card=4, 
                                    values=[
                                        [0.25, 0.25, 0.25], 
                                        [0.25, 0.25, 0.25], 
                                        [0.25, 0.25, 0.25], 
                                        [0.25, 0.25, 0.25]
                                    ], 
                                    evidence=['Department'], 
                                    evidence_card=[3])
        self.cpd_difficulty_level = TabularCPD(variable='Difficulty Level', variable_card=4, 
                                       values=[
                                           [0.1]*40,  # Simplified probability values; adjust as needed
                                           [0.2]*40,
                                           [0.3]*40,
                                           [0.4]*40
                                       ], 
                                       evidence=['Student Background', 'Type of Elective', 'Term Offered'], 
                                       evidence_card=[5, 4, 2])
        self.cpd_elective_subjects = TabularCPD(variable='Elective Subjects', variable_card=12, 
                                        values=np.random.dirichlet(np.ones(12), size=96).T,  # Generates random probabilities
                                        evidence=['Difficulty Level', 'Professor Names', 'Preferred Learning Mode'], 
                                        evidence_card=[4, 6, 4])
        self.cpd_preferred_learning_mode = TabularCPD(variable='Preferred Learning Mode', variable_card=4, 
                                                      values=np.random.dirichlet(np.ones(4), size=4).T, 
                                                      evidence=['Preferred Branch'], 
                                                      evidence_card=[4])
        self.cpd_preferred_branch = TabularCPD(variable='Preferred Branch', variable_card=4, 
                                               values=[[0.25], [0.25], [0.25], [0.25]])
        self.cpd_term_offered = TabularCPD(variable='Term Offered', variable_card=2, 
                                           values=[[0.5], [0.5]])

        # Add CPDs to the model
        self.model.add_cpds(
            self.cpd_student_background, self.cpd_location_university, self.cpd_department, 
            self.cpd_professor_names, self.cpd_type_elective, self.cpd_difficulty_level, self.cpd_elective_subjects, 
            self.cpd_preferred_learning_mode, self.cpd_preferred_branch, 
            self.cpd_term_offered
        )

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
            'Difficulty Level': int(user_input['Difficulty Level']),
            'Preferred Branch': int(user_input['Preferred Branch']),
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