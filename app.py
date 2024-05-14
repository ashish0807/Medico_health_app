from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import json
import os
from uuid import uuid4
from datetime import datetime


# flask app
app = Flask(__name__)


#datasets
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv("datasets/medications.csv")
diets = pd.read_csv("datasets/diets.csv")


svc = pickle.load(open("models/svc.pkl", "rb"))


# Get current date
current_date = datetime.now()

# Format the date
formatted_date = current_date.strftime("%d %B %Y")



def helper(dis):
    desc = description[description["Disease"] == dis]["Description"]
    desc = " ".join([w for w in desc])

    pre = precautions[precautions["Disease"] == dis][
        ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]
    ]
    pre = [col for col in pre.values]

    med = medications[medications["Disease"] == dis]["Medication"]
    med = [med for med in med.values]

    die = diets[diets["Disease"] == dis]["Diet"]
    die = [die for die in die.values]

    wrkout = workout[workout["disease"] == dis]["workout"]

    return desc, pre, med, die, wrkout

    
symptoms_dict = {
    "itching": 0,
    "skin_rash": 1,
    "nodal_skin_eruptions": 2,
    "continuous_sneezing": 3,
    "shivering": 4,
    "chills": 5,
    "joint_pain": 6,
    "stomach_pain": 7,
    "acidity": 8,
    "ulcers_on_tongue": 9,
    "muscle_wasting": 10,
    "vomiting": 11,
    "burning_micturition": 12,
    "spotting_ urination": 13,
    "fatigue": 14,
    "weight_gain": 15,
    "anxiety": 16,
    "cold_hands_and_feets": 17,
    "mood_swings": 18,
    "weight_loss": 19,
    "restlessness": 20,
    "lethargy": 21,
    "patches_in_throat": 22,
    "irregular_sugar_level": 23,
    "cough": 24,
    "high_fever": 25,
    "sunken_eyes": 26,
    "breathlessness": 27,
    "sweating": 28,
    "dehydration": 29,
    "indigestion": 30,
    "headache": 31,
    "yellowish_skin": 32,
    "dark_urine": 33,
    "nausea": 34,
    "loss_of_appetite": 35,
    "pain_behind_the_eyes": 36,
    "back_pain": 37,
    "constipation": 38,
    "abdominal_pain": 39,
    "diarrhoea": 40,
    "mild_fever": 41,
    "yellow_urine": 42,
    "yellowing_of_eyes": 43,
    "acute_liver_failure": 44,
    "fluid_overload": 45,
    "swelling_of_stomach": 46,
    "swelled_lymph_nodes": 47,
    "malaise": 48,
    "blurred_and_distorted_vision": 49,
    "phlegm": 50,
    "throat_irritation": 51,
    "redness_of_eyes": 52,
    "sinus_pressure": 53,
    "runny_nose": 54,
    "congestion": 55,
    "chest_pain": 56,
    "weakness_in_limbs": 57,
    "fast_heart_rate": 58,
    "pain_during_bowel_movements": 59,
    "pain_in_anal_region": 60,
    "bloody_stool": 61,
    "irritation_in_anus": 62,
    "neck_pain": 63,
    "dizziness": 64,
    "cramps": 65,
    "bruising": 66,
    "obesity": 67,
    "swollen_legs": 68,
    "swollen_blood_vessels": 69,
    "puffy_face_and_eyes": 70,
    "enlarged_thyroid": 71,
    "brittle_nails": 72,
    "swollen_extremeties": 73,
    "excessive_hunger": 74,
    "extra_marital_contacts": 75,
    "drying_and_tingling_lips": 76,
    "slurred_speech": 77,
    "knee_pain": 78,
    "hip_joint_pain": 79,
    "muscle_weakness": 80,
    "stiff_neck": 81,
    "swelling_joints": 82,
    "movement_stiffness": 83,
    "spinning_movements": 84,
    "loss_of_balance": 85,
    "unsteadiness": 86,
    "weakness_of_one_body_side": 87,
    "loss_of_smell": 88,
    "bladder_discomfort": 89,
    "foul_smell_of urine": 90,
    "continuous_feel_of_urine": 91,
    "passage_of_gases": 92,
    "internal_itching": 93,
    "toxic_look_(typhos)": 94,
    "depression": 95,
    "irritability": 96,
    "muscle_pain": 97,
    "altered_sensorium": 98,
    "red_spots_over_body": 99,
    "belly_pain": 100,
    "abnormal_menstruation": 101,
    "dischromic _patches": 102,
    "watering_from_eyes": 103,
    "increased_appetite": 104,
    "polyuria": 105,
    "family_history": 106,
    "mucoid_sputum": 107,
    "rusty_sputum": 108,
    "lack_of_concentration": 109,
    "visual_disturbances": 110,
    "receiving_blood_transfusion": 111,
    "receiving_unsterile_injections": 112,
    "coma": 113,
    "stomach_bleeding": 114,
    "distention_of_abdomen": 115,
    "history_of_alcohol_consumption": 116,
    "fluid_overload.1": 117,
    "blood_in_sputum": 118,
    "prominent_veins_on_calf": 119,
    "palpitations": 120,
    "painful_walking": 121,
    "pus_filled_pimples": 122,
    "blackheads": 123,
    "scurring": 124,
    "skin_peeling": 125,
    "silver_like_dusting": 126,
    "small_dents_in_nails": 127,
    "inflammatory_nails": 128,
    "blister": 129,
    "red_sore_around_nose": 130,
    "yellow_crust_ooze": 131,
}
diseases_list = {
    15: "Fungal infection",
    4: "Allergy",
    16: "GERD",
    9: "Chronic cholestasis",
    14: "Drug Reaction",
    33: "Peptic ulcer diseae",
    1: "AIDS",
    12: "Diabetes ",
    17: "Gastroenteritis",
    6: "Bronchial Asthma",
    23: "Hypertension ",
    30: "Migraine",
    7: "Cervical spondylosis",
    32: "Paralysis (brain hemorrhage)",
    28: "Jaundice",
    29: "Malaria",
    8: "Chicken pox",
    11: "Dengue",
    37: "Typhoid",
    40: "hepatitis A",
    19: "Hepatitis B",
    20: "Hepatitis C",
    21: "Hepatitis D",
    22: "Hepatitis E",
    3: "Alcoholic hepatitis",
    36: "Tuberculosis",
    10: "Common Cold",
    34: "Pneumonia",
    13: "Dimorphic hemmorhoids(piles)",
    18: "Heart attack",
    39: "Varicose veins",
    26: "Hypothyroidism",
    24: "Hyperthyroidism",
    25: "Hypoglycemia",
    31: "Osteoarthristis",
    5: "Arthritis",
    0: "(vertigo) Paroymsal  Positional Vertigo",
    2: "Acne",
    38: "Urinary tract infection",
    35: "Psoriasis",
    27: "Impetigo",
}
psychology_mcq = {
    1: {
        "question": "Overall how would you rate your physical health?",
        "options": ["Excellent", "Average", "Poor", "Not sure"],
        "correct_answer": None, 
    },
    2: {
        "question": "Overall how would you rate your mental health?",
        "options": ["Excellent", "Average", "Poor", "Not sure"],
        "correct_answer": None,
    },
    3: {
        "question": "During the past 3 weeks, how often has your mental health affected your ability to get work done?",
        "options": ["Excellent", "Average", "Poor", "Not sure"],
        "correct_answer": None,
    },
    4: {
        "question": "Have you felt particularly low or down for more than 2 weeks in a row?",
        "options": ["Yes", "No", "Not sure", "Prefer not to say"],
        "correct_answer": None,
    },
    5: {
        "question": "During the past two weeks, how often has your mental health affected your relationships?",
        "options": ["Excellent", "Average", "Poor", "Not sure"],
        "correct_answer": None,
    },
    6: {
        "question": "Have you noticed any change in your diet habits?",
        "options": ["Yes", "No", "Not sure", "Prefer not to say"],
        "correct_answer": None,
    },
    7: {
        "question": "When was the last time you were really happy?",
        "options": ["Few days ago", "Few weeks ago", "Few months ago", "Not sure"],
        "correct_answer": None,
    },
    8: {
        "question": "When was the last time you felt good about yourself?",
        "options": ["Few days ago", "Few weeks ago", "Few months ago", "Not sure"],
        "correct_answer": None,
    },
    9: {
        "question": "How often do you feel positive about your life?",
        "options": ["Never", "Once in a while", "About half the time", "Not sure"],
        "correct_answer": None,
    },
    10: {
        "question": "Have you ever been diagnosed with a mental disorder before?",
        "options": ["Yes", "No", "Not sure", "Prefer not to say"],
        "correct_answer": None,
    },
    11: {
        "question": "When did you last get your mental health examination done?",
        "options": ["Less than 2 weeks", "Less than 1 month", "Less than 6 months", "Less than 1 year"],
        "correct_answer": None,
    },
    12: {
        "question": "How many hours do you sleep per day?",
        "options": ["Less than 4 hours", "4-6 hours", "7-9 hours", "More than 9 hours"],
        "correct_answer": None,
    },
    13: {
        "question": "How is your quality of sleep?",
        "options": ["Very bad", "Bad", "Normal", "Good"],
        "correct_answer": None,
    },
    14: {
        "question": "What is your relationship status?",
        "options": ["Single", "Married", "It's complicated", "Prefer not to say"],
        "correct_answer": None,
    },
    15: {
        "question": "How often do you smoke?",
        "options": ["Never", "Once in a few weeks", "Once everyday", "More than once everyday"],
        "correct_answer": None,
    },
    16: {
        "question": "How often do you drink?",
        "options": ["Never", "Once in a few weeks", "Once everyday", "More than once everyday"],
        "correct_answer": None,
    },
    17: {
        "question": "When was the last time you had a positive outlook on life?",
        "options": ["Never", "Once in a few weeks", "Once everyday", "More than once everyday"],
        "correct_answer": None,
    },
    18: {
        "question": "Are you going through a tough emotional situation?",
        "options": ["Yes", "No", "Not sure", "Prefer not to say"],
        "correct_answer": None,
    },
}



# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]


# creating routes

@app.route("/")
def index():
    return render_template("landing.html")


# Define a route for the home page
@app.route("/predict", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        symptoms = request.form.get("symptoms")
      
        symptoms_dict_card = list(symptoms_dict.keys())
        symptoms_dict_card = np.random.choice(symptoms_dict_card, 5)

        print(symptoms)
        if symptoms == "Symptoms":
            message = (
                "Please enter the symptoms you are experiencing separated by commas."
            )
            return render_template(
                "predict.html",
                message=message,
                symptoms_dict=symptoms_dict,
                symptoms_dict_card=symptoms_dict_card,
            )
        else:

            user_symptoms = [s.strip() for s in symptoms.split(",")]
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(
                predicted_disease
            )
            rec_diet = eval(rec_diet[0])
            medications = eval(medications[0])

            my_precautions = []
            for i in precautions[0]:
                my_precautions.append(i)

            return render_template(
                "predict.html",
                predicted_disease=predicted_disease,
                dis_des=dis_des,
                my_precautions=my_precautions,
                medications=medications,
                my_diet=rec_diet,
                workout=workout,
                symptoms_dict=symptoms_dict,
                symptoms_dict_card=symptoms_dict_card,
                symptoms=symptoms,
            )
            # send random 5 symptoms to the index page

    symptoms_dict_card = list(symptoms_dict.keys())
    symptoms_dict_card = np.random.choice(symptoms_dict_card, 5)

    return render_template(
        "predict.html",
        symptoms_dict=symptoms_dict,
        symptoms_dict_card=symptoms_dict_card,
    )


# about route
@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/test", methods=["GET", "POST"])
def test():
    if request.method == "POST":
        # Generate unique UID for the session
        session_uid = str(uuid4())

        # Get user name and email from form
        user_name = request.form.get("name")
        user_email = request.form.get("email")

        # Load existing user data from JSON file
        users = []
        if os.path.exists("users.json"):
            with open("users.json", "r") as file:
                users = json.load(file)

        # Check if the user with the same email already exists
        user_exists = False
        for user_data in users:
            if user_data["email"] == user_email:
                user_exists = True
                user_data["totaltesttaken"] += 1
                break

        if not user_exists:
            # If user doesn't exist, create new user data
            # get current date 
            
            new_user_data = {
                "uid": session_uid,
                "name": user_name,
                "email": user_email,
                "date": formatted_date,  
                "totaltesttaken": 1,
            }
            users.append(new_user_data)

        # Save updated user data back to JSON file
        with open("users.json", "w") as file:
            json.dump(users, file, indent=4)



        # Calculate score and percentage
        score = 0
        total_questions = len(psychology_mcq)
        for question_number, question_data in psychology_mcq.items():
            user_answer = request.form.get("question" + str(question_number))
            correct_answer = question_data["correct_answer"]
            if user_answer == correct_answer:
                score += 1
        percentage_score = (score / total_questions) * 100
        print("Percentage Score:", percentage_score)



        # Return test results template with user email and ID
        return render_template(
            "ptest.html",
            score=score,
            total_questions=total_questions,
            percentage_score=percentage_score,
            psychology_mcq=psychology_mcq,
            user_email=user_email,
            user_name=user_name,
            session_uid=session_uid,
        )
    else:
        return render_template("ptest.html", psychology_mcq=psychology_mcq)


# create a route users and send  the  json file data to users 
@app.route("/users")
def users():
    users = []
    if os.path.exists("users.json"):
        with open("users.json", "r") as file:
            users = json.load(file)
    return render_template("users.html", users=users)

if __name__ == "__main__":
    app.run(debug=True, port=0)
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=8080)
