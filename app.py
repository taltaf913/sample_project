import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model/model.pkl')

# Streamlit application title
st.title('Cab Fare Prediction Model')

# Getting user input
st.write('Please enter the features to predict the cab fare:')

feature_1 = st.number_input('Feature 1', min_value=0.0, max_value=100.0, value=0.0)
feature_2 = st.number_input('Feature 2', min_value=0.0, max_value=100.0, value=0.0)

# Prediction
if st.button('Predict'):
    input_data = pd.DataFrame([[feature_1, feature_2]], columns=['Feature_1', 'Feature_2'])
    prediction = model.predict(input_data)
    #st.write(f'The predicted fare is: ${prediction[0]:.2f}')
    st.write(f'The predicted fare is: <span style="color: #008000;">${prediction[0]:.2f}</span>', unsafe_allow_html=True)

