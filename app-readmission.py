import numpy as np
import pickle
import pandas as pd
import streamlit as st 



pickle_in = open("trained_model_RForest2.pkl","rb")
price_finder=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def hospital_Readmission_price_predictor(encounter_id, patient_nbr, race, gender, age, weight, admission_type_id, discharge_disposition_id,
                                   admission_source_id, time_in_hospital, payer_code, medical_specialty, num_lab_procedures,
                                   num_procedures, num_medications, number_outpatient, number_emergency, number_inpatient,
                                   diag_1, diag_2, diag_3, number_diagnoses, max_glu_serum, A1Cresult, metformin, repaglinide,
                                   nateglinide, chlorpropamide, glimepiride, acetohexamide, glipizide, glyburide, tolbutamide,
                                   pioglitazone, rosiglitazone, acarbose, miglitol, troglitazone, tolazamide, examide, citoglipton,
                                   insulin, glyburide_metformin, glipizide_metformin, glimepiride_pioglitazone, metformin_rosiglitazone,
                                   metformin_pioglitazone, change, diabetesMed):
    
    
   
    prediction=price_finder.predict([[encounter_id, patient_nbr, race, gender, age, weight, admission_type_id, discharge_disposition_id,
                                   admission_source_id, time_in_hospital, payer_code, medical_specialty, num_lab_procedures,
                                   num_procedures, num_medications, number_outpatient, number_emergency, number_inpatient,
                                   diag_1, diag_2, diag_3, number_diagnoses, max_glu_serum, A1Cresult, metformin, repaglinide,
                                   nateglinide, chlorpropamide, glimepiride, acetohexamide, glipizide, glyburide, tolbutamide,
                                   pioglitazone, rosiglitazone, acarbose, miglitol, troglitazone, tolazamide, examide, citoglipton,
                                   insulin, glyburide_metformin, glipizide_metformin, glimepiride_pioglitazone, metformin_rosiglitazone,
                                   metformin_pioglitazone, change, diabetesMed]])
    print(prediction)
    return prediction



def main():
    st.title("hospital_Readmission  PREDICTOR")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit hospital_Readmission  Prediction  ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    encounter_id = st.text_input('Encounter ID', '2278392')
    patient_nbr = st.text_input('Patient Number', '8222157')
    race = st.selectbox('Race', ['Caucasian', 'AfricanAmerican', 'Other'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.selectbox('Age', ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])
    weight = st.text_input('Weight', '?')
    admission_type_id = st.number_input('Admission Type ID', value=6)
    discharge_disposition_id = st.number_input('Discharge Disposition ID', value=25)
    admission_source_id = st.number_input('Admission Source ID', value=1)
    time_in_hospital = st.number_input('Time in Hospital', value=1)
    payer_code = st.text_input('Payer Code', '?')
    medical_specialty = st.text_input('Medical Specialty', 'Pediatrics-Endocrinology')
    num_lab_procedures = st.number_input('Number of Lab Procedures', value=41)
    num_procedures = st.number_input('Number of Procedures', value=0)
    num_medications = st.number_input('Number of Medications', value=1)
    number_outpatient = st.number_input('Number of Outpatient Visits', value=0)
    number_emergency = st.number_input('Number of Emergency Visits', value=0)
    number_inpatient = st.number_input('Number of Inpatient Visits', value=0)
    diag_1 = st.text_input('Diagnosis 1', '250.83')
    diag_2 = st.text_input('Diagnosis 2', '?')
    diag_3 = st.text_input('Diagnosis 3', '?')
    number_diagnoses = st.number_input('Number of Diagnoses', value=1)
    max_glu_serum = st.text_input('Max Glucose Serum', 'None')
    A1Cresult = st.text_input('A1C Result', 'None')
    metformin = st.selectbox('Metformin', ['No', 'Yes'])
    repaglinide = st.selectbox('Repaglinide', ['No', 'Yes'])
    nateglinide = st.selectbox('Nateglinide', ['No', 'Yes'])
    chlorpropamide = st.selectbox('Chlorpropamide', ['No', 'Yes'])
    glimepiride = st.selectbox('Glimepiride', ['No', 'Yes'])
    acetohexamide = st.selectbox('Acetohexamide', ['No', 'Yes'])
    glipizide = st.selectbox('Glipizide', ['No', 'Yes'])
    glyburide = st.selectbox('Glyburide', ['No', 'Yes'])
    tolbutamide = st.selectbox('Tolbutamide', ['No', 'Yes'])
    pioglitazone = st.selectbox('Pioglitazone', ['No', 'Yes'])
    rosiglitazone = st.selectbox('Rosiglitazone', ['No', 'Yes'])
    acarbose = st.selectbox('Acarbose', ['No', 'Yes'])
    miglitol = st.selectbox('Miglitol', ['No', 'Yes'])
    troglitazone = st.selectbox('Troglitazone', ['No', 'Yes'])
    tolazamide = st.selectbox('Tolazamide', ['No', 'Yes'])
    examide = st.selectbox('Examide', ['No', 'Yes'])
    citoglipton = st.selectbox('Citoglipton', ['No', 'Yes'])
    insulin = st.selectbox('Insulin', ['No', 'Yes'])
    glyburide_metformin = st.selectbox('Glyburide-Metformin', ['No', 'Yes'])
    glipizide_metformin = st.selectbox('Glipizide-Metformin', ['No', 'Yes'])
    glimepiride_pioglitazone = st.selectbox('Glimepiride-Pioglitazone', ['No', 'Yes'])
    metformin_rosiglitazone = st.selectbox('Metformin-Rosiglitazone', ['No', 'Yes'])
    metformin_pioglitazone = st.selectbox('Metformin-Pioglitazone', ['No', 'Yes'])
    change = st.selectbox('Change of Medications', ['No', 'Ch'])
    diabetesMed = st.selectbox('Diabetes Medication', ['No', 'Yes'])
    # readmitted = st.selectbox('Readmitted', ['NO', '>30', '<30'])


    result=None
    if st.button("Predict"):
        result=hospital_Readmission_price_predictor(encounter_id, patient_nbr, race, gender, age, weight, admission_type_id, discharge_disposition_id,
                                   admission_source_id, time_in_hospital, payer_code, medical_specialty, num_lab_procedures,
                                   num_procedures, num_medications, number_outpatient, number_emergency, number_inpatient,
                                   diag_1, diag_2, diag_3, number_diagnoses, max_glu_serum, A1Cresult, metformin, repaglinide,
                                   nateglinide, chlorpropamide, glimepiride, acetohexamide, glipizide, glyburide, tolbutamide,
                                   pioglitazone, rosiglitazone, acarbose, miglitol, troglitazone, tolazamide, examide, citoglipton,
                                   insulin, glyburide_metformin, glipizide_metformin, glimepiride_pioglitazone, metformin_rosiglitazone,
                                   metformin_pioglitazone, change, diabetesMed)
    st.success('The predicted readmission status is: {}'.format(result))

    
if __name__=='__main__':
    main()
    
