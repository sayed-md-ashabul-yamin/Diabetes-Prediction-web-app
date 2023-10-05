# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:01:13 2023

@author: ADMIN
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


def diabetes_prediction(input_data):
    
    input_data = (4,110,92,0,0,37.6,0.191,30)

    # changing the data into a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return "The Person is Non Diabetic"
    else:
      return "The Person is Diabetic"
  
    
  
    
def main():
      
    #giving a title
    st.title("Diabetes Prediction Model")
    
    #getting the input data from the user
    
   # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    
    
    Pregnancies = st.text_input("Number of Pregnancies")
    
    Glucose = st.text_input("Glucose Level")
    
    BloodPressure = st.text_input("Blood Pressure Value")
    
    SkinThickness = st.text_input("Skin Thickness Value")
    
    Insulin = st.text_input("Insulin Level")
    
    BMI = st.text_input("BMI Value")
    
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    
    Age = st.text_input("Age of The Person")
    
    
    # code for prediction
    
    diagnosis = ""
    
    # creating the button for prediction
    
    if st.button("Diabetes Test Result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    st.success(diagnosis)
    

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
