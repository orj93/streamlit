import streamlit as st
import pickle
import numpy as np



with (open(r"C:\Users\user\4Geeks\Módulos\Módulo25\streamlit\models\tree_model.pkl", "rb")) as openfile:
     model = pickle.load(openfile)

# with (open(r"..\models\tree_model.pkl", "rb")) as openfile:
#      model = pickle.load(openfile)

diabetes_dict = {
    0:"No",
    1:"Yes"
}

st.title("Diabetes predictor")

number_of_pregnancies = st.number_input(
    "Enter the number of pregnancies")

glucose_value = st.number_input(
    'Insert the glucose value')

blood_bressure = st.number_input(
    "Enter your blood pressure")

skin_thickness = st.number_input(
    'Enter your skin thickness')

insuline = st.number_input(
    "Enter the value of insuline")

bmi = st.number_input(
    'Insert your BMI')

ped_function = st.number_input(
    "Enter your diabetes pedigree function")

age = st.number_input(
    'Enter your age')

if st.button("Predecir"):
    row = [
        number_of_pregnancies,
        glucose_value,
        blood_bressure,
        skin_thickness,
        insuline,
        bmi,
        ped_function,
        age
    ]
    array=np.array(row)
    
    row_reshaped = array.reshape(1, -1)    
    y_pred = model.predict(row_reshaped)[0]

#    st.text(str(y_pred))

    pred_class = diabetes_dict[y_pred]
    st.write("Prediction:", pred_class)





 