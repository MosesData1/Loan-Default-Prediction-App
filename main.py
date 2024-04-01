import streamlit as st
import pandas as pd 
import numpy as np 
from joblib import load 
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder


# load the saved model 
model = load('loan_models') 



def preprocess_data(Age, Income, Loan_Amount, Credit_Score, Months_Employed, Num_Credit_Lines, Interest_Rate, Loan_Term,
                    DTI_Ratio, Education, Employment_Type, Marital_Status, Has_Mortgage, Has_Dependents, Loan_Purpose, Has_CoSigner):

    Data = ({
    'Age': [Age],
    'Income': [Income], 
    'Loan_Amount': [Loan_Amount], 
    'Credit_Score': [Credit_Score], 
    'Months_Employed': [Months_Employed], 
    'Num_Credit_Lines': [Num_Credit_Lines], 
    'Interest_Rate': [Interest_Rate], 
    'Loan_Term': [Loan_Term],
    'DTI_Ratio': [DTI_Ratio], 
    'Education': [Education], 
    'Employment_Type': [Employment_Type], 
    'Marital_Status': [Marital_Status], 
    'Has_Mortgage': [Has_Mortgage], 
    'Has_Dependents': [Has_Dependents], 
    'Loan_Purpose': [Loan_Purpose], 
    'Has_CoSigner': [Has_CoSigner]
    })

    df = pd.DataFrame(Data)


    # encoding Binary Data
    df['Has_Mortgage'] = df['Has_Mortgage'].map({'Yes': 1, 'No': 0})
    df['Has_Dependents'] = df['Has_Dependents'].map({'Yes': 1, 'No': 0})
    df['Has_CoSigner'] = df['Has_CoSigner'].map({'Yes': 1, 'No': 0})

    #encoding categorical data
    label_encoder = LabelEncoder()
    df['Education'] = label_encoder.fit_transform(df['Education'])
    df['Employment_Type'] = label_encoder.fit_transform(df['Employment_Type'])
    df['Marital_Status'] = label_encoder.fit_transform(df['Marital_Status'])
    df['Loan_Purpose'] = label_encoder.fit_transform(df['Loan_Purpose'])


    # Let Scale our data
    Scaler = MinMaxScaler()
    df = Scaler.fit_transform(df)

    return df



def main():
    st.title('Loan Default Prediction App') 
    st.write('This is a loan default prediction app, Please enter the reqiured information to get the prediction')
    st.image('loan_image.jpeg', use_column_width=True)
    st.subheader('Please enter the required information')
    st.divider()
    
    # Get user input
    Age = st.slider('Age', min_value=18, max_value=100, value=35)
    Income = st.number_input('Income', value=100000)
    Loan_Amount = st.number_input('Loan Amount', value=100000)
    Credit_Score = st.slider('Credit Score', min_value=300, max_value=850, value=650)
    Months_Employed = st.number_input('Months Employed', value=12)
    Num_Credit_Lines = st.number_input('Number of Credit Lines', value=2)
    Interest_Rate = st.slider('Interest Rate', min_value=0, max_value=30, value=15)
    Loan_Term = st.slider('Loan Term (months)', min_value=12, max_value=60, value=36)
    DTI_Ratio = st.slider('DTI Ratio', min_value=0.0, max_value=1.0, value=0.5)
    Education = st.selectbox("Education",["High School", "Bachelor's", "Master's", "PhD"])
    Employment_Type = st.selectbox("Employment Type", ["Full time", "Part time", "Self-employed", "Unemployed"])
    Marital_Status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    Has_Mortgage = st.radio("Has Mortgage", ["Yes", "No"])
    Has_Dependents = st.radio("Has Dependents", ["Yes", "No"])
    Loan_Purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])
    Has_CoSigner = st.radio("Has Co-signer", ["Yes", "No"])

    # preprocess the user input
    user_data = preprocess_data(Age, Income, Loan_Amount, Credit_Score, Months_Employed, Num_Credit_Lines, Interest_Rate, Loan_Term,
                    DTI_Ratio, Education, Employment_Type, Marital_Status, Has_Mortgage, Has_Dependents, Loan_Purpose, Has_CoSigner)

    # make prediction with the loaded model
    prediction = model.predict(user_data)
    
    # Display the prediction
    st.subheader('Prediction')
    st.write("1: Loan will default, 0: Loan will not default")
    result = 'loan will default' if prediction[0] == 1 else 'loan will not default'
    st.write(prediction[0], result)


if __name__ == "__main__":
    main()





