import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Set page config
st.set_page_config(
    page_title="Medical Care Diagnosis System",
    page_icon="🏥",
    layout="wide"
)

# Load datasets
@st.cache_data
def load_data():
    sym_des = pd.read_csv("datasets/symtoms_df.csv")
    precautions = pd.read_csv("datasets/precautions_df.csv")
    workout = pd.read_csv("datasets/workout_df.csv")
    description = pd.read_csv("datasets/description.csv")
    medications = pd.read_csv('datasets/medications.csv')
    diets = pd.read_csv("datasets/diets.csv")
    return sym_des, precautions, workout, description, medications, diets

sym_des, precautions, workout, description, medications, diets = load_data()

# Load model
@st.cache_resource
def load_model():
    return pickle.load(open('models/svc.pkl', 'rb'))

svc = load_model()

# Helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']

    return desc, pre, med, die, wrkout

# Symptoms and diseases dictionaries
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 
                'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 
                'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 
                'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 
                'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 
                'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 
                'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 
                'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 
                'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 
                'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 
                'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 
                'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 
                'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 
                'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 
                'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 
                'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 
                'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 
                'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 
                'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 
                'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 
                'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 
                'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 
                'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 
                'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 
                'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 
                'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 
                'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 
                14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 
                17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 
                7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 
                29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 
                19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 
                3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 
                13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 
                26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 
                5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 
                38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# Main app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "About", "Contact", "Developer", "Blog"])

    if page == "Home":
        home_page()
    elif page == "About":
        about_page()
    elif page == "Contact":
        contact_page()
    elif page == "Developer":
        developer_page()
    elif page == "Blog":
        blog_page()

def home_page():
    st.title("Medical Care Diagnosis System")
    st.write("Enter your symptoms below to get a diagnosis and recommendations.")

    # Input symptoms
    symptoms = st.text_input("Enter your symptoms (comma separated):", 
                            placeholder="e.g., headache, fever, cough")
    
    if st.button("Predict"):
        if not symptoms or symptoms == "Symptoms":
            st.warning("Please enter valid symptoms (comma separated)")
        else:
            # Process symptoms
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            
            try:
                predicted_disease = get_predicted_value(user_symptoms)
                dis_des, prec, med, rec_diet, wrkout = helper(predicted_disease)
                
                st.success(f"Predicted Disease: {predicted_disease}")
                
                # Display results in tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Description", "Precautions", "Medications", "Diet", "Workout"])
                
                with tab1:
                    st.subheader("Description")
                    st.write(dis_des)
                
                with tab2:
                    st.subheader("Precautions")
                    for i, precaution in enumerate(prec[0], 1):
                        st.write(f"{i}. {precaution}")
                
                with tab3:
                    st.subheader("Medications")
                    for medication in med:
                        st.write(f"- {medication}")
                
                with tab4:
                    st.subheader("Recommended Diet")
                    for diet in rec_diet:
                        st.write(f"- {diet}")
                
                with tab5:
                    st.subheader("Recommended Workout")
                    for exercise in wrkout:
                        st.write(f"- {exercise}")
                        
            except Exception as e:
                st.error(f"Error processing your symptoms. Please check your input and try again. Error: {str(e)}")

def about_page():
    st.title("About This Project")
    st.write("""
    This is a medical care diagnosis system that helps users identify potential diseases 
    based on their symptoms. The system uses machine learning to predict the most likely 
    condition and provides recommendations for precautions, medications, diet, and exercises.
    """)
    st.write("""
    The system is designed for educational purposes only and should not replace professional 
    medical advice. Always consult with a healthcare provider for proper diagnosis and treatment.
    """)

def contact_page():
    st.title("Contact Us")
    st.write("Have questions or feedback? Reach out to us!")
    st.write("Email: contact@medicalcaresystem.com")
    st.write("Phone: +1 (123) 456-7890")
    st.write("Address: 123 Medical Street, Health City, HC 12345")

def developer_page():
    st.title("Developer Information")
    st.write("This application was developed by:")
    st.write("- Your Name")
    st.write("- Your Team (if applicable)")
    st.write("For more information about the development process or to contribute, please visit our GitHub repository.")

def blog_page():
    st.title("Medical Care Blog")
    st.write("Coming soon! Articles and updates about medical care, health tips, and system improvements.")

if __name__ == '__main__':
    main()
