import streamlit as st
import pandas as pd
from pycaret.classification import predict_model,load_model

gbc = load_model('./gbc_churn_0125')

st.set_page_config(page_title='Bank Churn Analysis',
                   page_icon='🏦',
                   layout='wide',
                   initial_sidebar_state="expanded")

st.markdown('# Analise de Churn de Clientes Bancários')

st.markdown('## Implementação')

st.write('Upload um **CSV** ou preencha o **formulário**.')

with st.sidebar:
    df = st.file_uploader('Previsão com arquivo CSV:', type=['csv'])
    
    st.write('Previsão com formulário:')
    with st.form('Simulação'):
        credit_score = st.slider('Credit Score: ', min_value=350, max_value=850, step=1)
        geography = st.selectbox('Geography: ', options=['France', 'Germany', 'Spain'])
        gender = st.selectbox('Gender: ', options=['Female', 'Male'])
        age = st.number_input('Age: ', step=1, min_value=18, max_value=100)
        tenure = st.slider('Tenure: ', min_value=0, max_value=10, step=1)
        balance = st.number_input('Balance: ', step=10.0, min_value=0.0)
        num_of_products = st.slider('Number of Products: ', step=1, min_value=1, max_value=4)
        has_cr_card = st.checkbox('Has Credit Card')
        is_active_member = st.checkbox('Is Active Member')
        estimated_salary = st.number_input('Estimated Salary: ', step=10.0, min_value=0.0)
        submitted = st.form_submit_button('Enviar')

if df:
    df = pd.read_csv(df).drop(['Surname', 'CustomerId', 'RowNumber', 'Exited'], axis=1)

    pred = predict_model(gbc, df, probability_threshold=0.478)
    st.write(pred)

    csv = pred.to_csv(index=False)

    st.download_button(
        label="Baixar previsões 📩",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )

if submitted:
    st.write(
    predict_model(gbc, data=pd.DataFrame({
        "CreditScore":credit_score,
        "Geography":geography,
        "Gender":gender,
        "Age":age,
        "Tenure":tenure,
        "Balance":balance,
        "NumOfProducts":num_of_products,
        "HasCrCard":has_cr_card,
        "IsActiveMember":is_active_member,
        "EstimatedSalary":estimated_salary
        }, index=[0]), probability_threshold=0.481))