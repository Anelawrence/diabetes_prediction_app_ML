import numpy as np
import pickle
import streamlit as st


# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# loading the scaler object
scaler = pickle.load(open('scaler_object.sav', 'rb')) # C:/Users/LAWRENCE/Desktop/SGA_1.3/PROJECTS_Folder/


def diabetes_prediction(input_values):
    input_value_as_array = np.asarray(input_values)
    input_data_reshape = input_value_as_array.reshape(1,-1)
    # Standardizing the user input with standard scaler
    input_data_reshape = scaler.transform(input_data_reshape)
    predict_value = loaded_model.predict(input_data_reshape)
    print(predict_value)
    if predict_value[0]==0:
       return 'The person is not diabetic'
    else:
       return 'The person is diabetic'
    

def main():
   
    # giving a title
    st.title('Diabetes Prediction Web App')
    st.text_input('Enter your Name:', key='name')
    

    #Getting inpit data from user
    Pregnancies = st.number_input('Number of Pregnancies:')
    Glucose = st.number_input('Glucose Level:')
    BloodPressure = st.number_input('Blood Pressure value:')
    SkinThickness = st.number_input('Skin Thickness value:')
    Insulin = st.number_input('Insulin Level:')
    BMI = st.number_input('BMI value:')
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value:')
    Age = st.number_input('Age of the Person:')

    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        
    st.success(diagnosis)



if __name__== '__main__':
    main()


st.write(f"Stay healthy {st.session_state.name}")
    