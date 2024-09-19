import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import subprocess
import time
import os

# Suppress oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Function to start TensorBoard
def start_tensorboard(logdir):
    tensorboard_process = subprocess.Popen(
        ["tensorboard", "--logdir", logdir]
    )
    time.sleep(5)  # Wait for TensorBoard to start
    return tensorboard_process

# Set the path to your TensorBoard logs
log_directory = "logs/fit20240920-001317"  # Your specified log directory

# Start TensorBoard
tensorboard_process = start_tensorboard(log_directory)

# Load model and encoders
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit layout
st.set_page_config(layout="wide")  # Set wide layout

# Main app section
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')

# TensorBoard section below predictions
st.header("Training and Validation Graph Using TensorBoard")
st.markdown(
    f'<iframe src="http://127.0.0.1:6006" width="100%" height="600"></iframe>',
    unsafe_allow_html=True,
)

# Display CSV data below TensorBoard
st.header("Dataset Preview")
try:
    # Replace 'data.csv' with your actual CSV file path
    data = pd.read_csv(r'C:\Users\HP\Desktop\Generative AI\Regression Project with ANN\Churn_Modelling.csv')
    st.dataframe(data)  # Display the data as a table
except FileNotFoundError:
    st.error("The CSV file was not found. Please check the file path.")

# Optional: Stop TensorBoard when the Streamlit app stops
if st.button("Stop TensorBoard"):
    tensorboard_process.terminate()
    st.success("TensorBoard stopped.")
