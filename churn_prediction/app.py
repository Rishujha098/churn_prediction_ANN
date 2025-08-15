import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

#load the trained model
model = tf.keras.models.load_model("model.h5")

#load the encoders
with open("onehot_encodr_geo.pkl","rb") as file:
    onehot_encodr_geo=pickle.load(file)

with open("label_encodr_gender.pkl","rb") as file:
    label_encodr_gender=pickle.load(file)

with open("Scaler.pkl","rb") as file:
    Scaler=pickle.load(file)

#streamlit app
st.title("Customer churn prediction")     



# User input
geography = st.selectbox('Geography', onehot_encodr_geo.categories_[0])
gender = st.selectbox('Gender', label_encodr_gender.classes_)
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
    'Gender': [label_encodr_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode geography
geo_encoded = onehot_encodr_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encodr_geo.get_feature_names_out(["Geography"]))

# Concatenate
final_input = pd.concat([input_data, geo_encoded_df], axis=1)

# Scale
input_data_scaled = Scaler.transform(final_input)

# Predict
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(prediction_proba)

if prediction_proba < 0.5:
    st.write("Customer is going to leave")
else:
    st.write("Customer is not leaving")
