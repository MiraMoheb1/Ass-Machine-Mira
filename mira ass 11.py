import streamlit as st 
import pickle
import pandas as pd 
import sklearn

st.title('Heart Disease Prediction')
st.info('Heart disease prediction based on user input features')
st.sidebar.header('Feature Selection')

# Collecting user input for the features
age = st.text_input('Age')
sex = st.text_input('Sex')
cp = st.text_input('Chest Pain Type')
trestbps = st.text_input('Resting Blood Pressure')
chol = st.text_input('Serum Cholesterol Level')
fbs = st.text_input('Fasting Blood Sugar')
restecg = st.text_input('Resting Electrocardiographic Results')
thalach = st.text_input('Maximum Heart Rate Achieved')
exang = st.text_input('Exercise Induced Angina')
oldpeak = st.text_input('ST Depression Induced By Exercise Relative To Rest')
slope = st.text_input('Slope Of The Peak Exercise Segment')
ca = st.text_input('Number of Major Vessels Colored by Flouroscopy')
thal = st.text_input('Thalassemia Type')
total_chol = st.text_input('Total Cholesterol')
exang_thalach_interaction = st.text_input('Exercise Induced Angina * Maximum Heart Rate Achieved')
age_chol_interaction = st.text_input('Age * Serum Cholesterol Level')
thalach_exang_interaction = st.text_input('Maximum Heart Rate Achieved * Exercise Induced Angina')

# Creating DataFrame from user input
df = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal],
    'total_chol': [total_chol],
    'exang_thalach_interaction': [exang_thalach_interaction],
    'age_chol_interaction': [age_chol_interaction],
    'thalach_exang_interaction': [thalach_exang_interaction]
}, index=[0])

# Loading the pre-trained model
logistic_model=pickle.load(open(r"C:\Users\MIRA\Downloads\Heart Disease.sav",'rb'))

# Predicting the output
if st.button('Predict'):
    prediction = logistic_model.predict(df)
    st.write('Prediction:', 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease')
