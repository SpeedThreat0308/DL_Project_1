import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


model=load_model("model.h5")

with open("label_encoder.pkl","rb") as file:
    label=pickle.load(file)

with open("one_hot_encoded.pkl","rb") as file:
    one_hot=pickle.load(file)

with open("scaler.pkl","rb") as file:
    scaler=pickle.load(file)


#Streamlit title
st.title("Customer Churn Prediction")

#User Input
geography=st.selectbox("Geography",one_hot.categories_[0])
gender=st.selectbox("Gender",label.classes_)
age=st.slider("Age",18,92)
balance=st.number_input("Balance",)
credit_score=st.number_input("Credit_Score")
estimated_salary=st.number_input("Estimated_Salary")
tenure=st.slider("Tenure",0,10)
num_of_products=st.slider("Num_Of_products",1,4)
has_cr_card=st.selectbox("Has_Cr_Card",[0,1])
is_active_member=st.selectbox("Is_Active_Member",[0,1])

#Example New Test Data

final_gender = label.transform([gender])[0]

input_data={
    "CreditScore": [credit_score],
    "Gender": [final_gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
}

input_df=pd.DataFrame(input_data)

encoded=one_hot.transform([[geography]])
geo_encoded=pd.DataFrame(encoded.toarray(),columns=one_hot.get_feature_names_out(["Geography"]))

data=pd.concat([input_df.reset_index(drop=True),geo_encoded],axis=1)

data_scaled=scaler.transform(data)

prediction=model.predict(data_scaled)
prediction_proba=prediction[0][0]

st.write(f"Churn Probabiltiy: {prediction_proba:.2f}")

if prediction_proba>0.5:
    st.write("The customer will churn")

else: 
    st.write("The customer will not churn")