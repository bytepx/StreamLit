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
def hospital_Readmission_price_predictor(age, time_in_hospital, num_procedures, num_medications):
    
    
   
    prediction=price_finder.predict([[age, time_in_hospital, num_procedures, num_medications]])
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
    age = st.number_input('Age', value=10)
    time_in_hospital = st.number_input('Time in Hospital', value=1)
    num_procedures = st.number_input('Number of Procedures', value=0)
    num_medications = st.number_input('Number of Medications', value=1)
    



    result=None
    if st.button("Predict"):
        result=hospital_Readmission_price_predictor(age, time_in_hospital, num_procedures, num_medications)
    st.success('The predicted readmission status is: {}'.format(result))

    
if __name__=='__main__':
    main()
    
