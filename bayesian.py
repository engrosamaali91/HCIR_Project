import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the Bayesian Network structure
model = BayesianNetwork([
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
cpd_student_background = TabularCPD('Student Background', 5, [[0.2], [0.2], [0.2], [0.2], [0.2]])

cpd_location_university = TabularCPD('Location of University', 2, [[0.5], [0.5]])

cpd_department = TabularCPD('Department', 3, 
                            [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 
                             [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], 
                             [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]], 
                            evidence=['Student Background', 'Location of University'], 
                            evidence_card=[5, 2])

cpd_type_elective = TabularCPD('Type of Elective', 4, 
                               [[0.4, 0.3, 0.3],  # Department: CS
                                [0.3, 0.4, 0.3],  # EE
                                [0.2, 0.2, 0.4],  # ME
                                [0.1, 0.1, 0.0]], # Sums to 1
                               evidence=['Department'], 
                               evidence_card=[3])

cpd_course_attributes = TabularCPD('Course Attributes', 4, 
                                   np.random.dirichlet(np.ones(4), size=240).T, 
                                   evidence=['Department', 'Type of Elective', 'Professor Names', 'Term Offered'], 
                                   evidence_card=[3, 4, 10, 2])

cpd_elective_subjects = TabularCPD('Elective Subjects', 12, 
                                   np.random.dirichlet(np.ones(12), size=16).T, 
                                   evidence=['Course Attributes', 'Preferred Learning Mode'], 
                                   evidence_card=[4, 4])

cpd_preferred_learning_mode = TabularCPD('Preferred Learning Mode', 4, 
                                         np.random.dirichlet(np.ones(4), size=4).T, 
                                         evidence=['Student Preferences'], 
                                         evidence_card=[4])

cpd_student_preferences = TabularCPD('Student Preferences', 4, 
                                     [[0.25], [0.25], [0.25], [0.25]])

cpd_professor_names = TabularCPD('Professor Names', 10, 
                                 [[0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1]])
cpd_term_offered = TabularCPD('Term Offered', 2, [[0.5], [0.5]])

# Add CPDs to the model
model.add_cpds(cpd_student_background, cpd_location_university, cpd_department, 
               cpd_type_elective, cpd_course_attributes, cpd_elective_subjects, 
               cpd_preferred_learning_mode, cpd_student_preferences, 
               cpd_professor_names, cpd_term_offered)

# Validate the model
assert model.check_model()

# Inference
inference = VariableElimination(model)

def get_user_input():
    user_input = {}
    print("Choose from the following options:")
    print("Student Background Major (0: Computer Science, 1: Electrical Engineering, 2: Mechanical Engineering, 3: Mechatronics, 4: Autonomous Systems)")
    user_input['Student Background'] = input("Enter Student Background (0-4): ")
    
    print("Location of University (0: Campus Sankt Augustin, 1: Campus Rheinbach)")
    user_input['Location of University'] = input("Enter Location of University (0-1): ")
    
    print("Department (0: Computer Science, 1: Electrical Engineering, 2: Mechanical Engineering)")
    user_input['Department'] = input("Enter Department (0-2): ")
    
    print("Type of Elective (0: Core, 1: Specialization, 2: Research, 3: Interdisciplinary)")
    user_input['Type of Elective'] = input("Enter Type of Elective (0-3): ")
    
    print("Professor Names (0: Prof. Dr. Wolfgang Borutzky, 1: Prof. Dr. Alexander Asteroth, 2: Dr. Aleksandar Mitrevski, 3: Prof. Dr. Sebastian Houben, 4: Prof. Dr. Joern Hees, 5: Prof. Dr. Erwin Prassler, 6: Prof. Dr. Teena Hassan, 7: M.Sc. Tim Metzler, 8: M.Sc. Youssef Mahmoud Youssef, 9: Prof. Dr. Paul G. Plöger)")
    user_input['Professor Names'] = input("Enter Professor Names (0-9): ")
    
    print("Term Offered (0: Summer Semester, 1: Winter Semester)")
    user_input['Term Offered'] = input("Enter Term Offered (0-1): ")
    
    print("Student Preferences (0: AI, 1: Robotics, 2: Data Science, 3: Sustainable Energy)")
    user_input['Student Preferences'] = input("Enter Student Preferences (0-3): ")
    
    print("Preferred Learning Mode (0: Lecture, 1: Lab, 2: Seminar, 3: Workshop)")
    user_input['Preferred Learning Mode'] = input("Enter Preferred Learning Mode (0-3): ")
    
    return user_input

def recommend_electives(user_input):
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
    query_result = inference.query(variables=['Elective Subjects'], evidence=evidence)
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

if __name__ == "__main__":
    user_input = get_user_input()
    recommendations = recommend_electives(user_input)
    print("Top elective recommendations based on your preferences are:")
    for elective, probability in recommendations:
        print(f"{elective}: {probability:.4f}")