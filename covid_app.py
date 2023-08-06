import pandas as pd
import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder
import sklearn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Simple Web application Page")
tab_1, tab_2,tab_3 = st.tabs(['DATAFRAME AND DOWNLOAD','VIEW PREDICTION','ABOUT APPLICATION'])
model_dt = joblib.load("dt.joblib")
model_lr = pickle.load(open("lr.pkl", 'rb'))

tab_3.write("About The Application")

alg_option = st.sidebar.selectbox("Preferred Algorithm for prediction?",("Logistic Regression","Decision Tree"))

pred_option = st.sidebar.selectbox("Preferred prediction mode",("Single","Multiple"))

if pred_option.lower() == "single" :
    br = st.sidebar.selectbox("Do you have a breathing problem?",("Yes","No"))
    fv = st.sidebar.selectbox("Do you have a fever?",("Yes","No"))
    dc = st.sidebar.selectbox("Do you have a dry cough?",("Yes","No"))
    sr = st.sidebar.selectbox("Do you have a sore throat?",("Yes","No"))
    rn = st.sidebar.selectbox("Do you have runny nose?",("Yes","No"))
    at = st.sidebar.selectbox("Do you have Asthma?",("Yes","No"))
    cld = st.sidebar.selectbox("Do you have Chronic Lung Disease?",("Yes","No"))
    h = st.sidebar.selectbox("Do you have headache?",("Yes","No"))
    hd = st.sidebar.selectbox("Do you have Heart Disease?",("Yes","No"))
    db = st.sidebar.selectbox("Do you have a Diabetes?",("Yes","No"))
    ht = st.sidebar.selectbox("Do you have a Hyper Tension?",("Yes","No"))
    fg = st.sidebar.selectbox("Do you have a Fatigue?",("Yes","No"))
    gt = st.sidebar.selectbox("Do you have a Gastrointestinal?",("Yes","No"))
    at = st.sidebar.selectbox("Do you have a Abroad travel?",("Yes","No"))
    ccp = st.sidebar.selectbox("Do you have a contact with COVID Patient?",("Yes","No"))
    atg = st.sidebar.selectbox("Do you have a Attended Large Gathering?",("Yes","No"))
    vpep = st.sidebar.selectbox("Do you have a Visited Public Exposed Places?",("Yes","No"))
    fwpep = st.sidebar.selectbox("Do you have a Family working in Public Exposed Places?",("Yes","No"))
    wm = st.sidebar.selectbox("Do you wear masks?",("Yes","No"))
    sfm = st.sidebar.selectbox("Do you sanitize yourself from market?",("Yes","No"))

    df = pd.DataFrame()

    df["Breathing Problem"] = [br]
    df['Fever'] = [fv]
    df['Dry Cough'] = [dc]
    df['Sore throat'] = sr
    df['Running Nose'] = rn
    df['Asthma'] = at
    df['Chronic Lung Disease'] = cld
    df['Headache'] = h
    df['Heart Disease'] = hd
    df['Diabetes'] = db
    df['Hyper Tension'] = ht
    df['Fatigue '] = fg
    df['Gastrointestinal '] = gt
    df["Abroad travel"] = at
    df['Contact with COVID Patient'] = ccp
    df['Attended Large Gathering'] = atg
    df['Visited Public Exposed Places'] = vpep
    df['Family working in Public Exposed Places'] = fwpep

    X = ['Breathing Problem_No', 'Breathing Problem_Yes', 'Fever_No',
        'Fever_Yes', 'Dry Cough_No', 'Dry Cough_Yes', 'Sore throat_No',
        'Sore throat_Yes', 'Running Nose_No', 'Running Nose_Yes', 'Asthma_No',
        'Asthma_Yes', 'Chronic Lung Disease_No', 'Chronic Lung Disease_Yes',
        'Headache_No', 'Headache_Yes', 'Heart Disease_No', 'Heart Disease_Yes',
        'Diabetes_No', 'Diabetes_Yes', 'Hyper Tension_No', 'Hyper Tension_Yes',
        'Fatigue _No', 'Fatigue _Yes', 'Gastrointestinal _No',
        'Gastrointestinal _Yes', 'Abroad travel_No', 'Abroad travel_Yes',
        'Contact with COVID Patient_No', 'Contact with COVID Patient_Yes',
        'Attended Large Gathering_No', 'Attended Large Gathering_Yes',
        'Visited Public Exposed Places_No', 'Visited Public Exposed Places_Yes',
        'Family working in Public Exposed Places_No',
        'Family working in Public Exposed Places_Yes']

    df_dummies = pd.get_dummies(df)

    df_encode = pd.DataFrame()
    for i in (X) : 
        if i in (df_dummies) :
            df_encode[i] = [1]
        else :
            df_encode[i] = [0]


    tab_1.dataframe(df)

    tab_2.dataframe(df_encode)

    pred_lr = model_lr.predict(df_encode)
    pred_dt = model_dt.predict(df_encode)

    if alg_option.lower() == "logistic regression" :

        tab_2.title("Logistic Regression prediction")
        if pred_lr == "Yes" :
            tab_1.title("From the features provided, The patient is predicted to be COVID-19 POSITIVE")
        else :
            tab_1.title("From the features provided, The patient is predicted to be COVID-19 NEGATIVE")
        tab_2.write(f'probability of having covid is {model_lr.predict_proba(df_encode)[:,1] * 100} %')

        proba_lr = model_lr.predict_proba(df_encode)

        fig = sns.barplot(x=np.arange(len(proba_lr[0])), y=proba_lr[0])
        plt.xticks(np.arange(len(proba_lr[0])), labels=[f"Class {i}" for i in range(len(proba_lr[0]))])
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title(f'Predicted Probabilities for Single Sample')
        plt.savefig('predicted_probabilities.png')
        tab_1.pyplot()
    else :
        tab_2.title("Decision Tree prediction")
        if pred_dt == "Yes" :
            tab_1.title("From the features provided, The patient is predicted to be COVID-19 POSITIVE")
        else :
            tab_1.title("From the features provided, The patient is predicted to be COVID-19 NEGATIVE")
        tab_2.write(f'probability of having covid is {model_dt.predict_proba(df_encode)[:,1] * 100} % ')

        proba_dt = model_dt.predict_proba(df_encode)

        fig = sns.barplot(x=np.arange(len(proba_dt[0])), y=proba_dt[0])
        plt.xticks(np.arange(len(proba_dt[0])), labels=[f"Class {i}" for i in range(len(proba_dt[0]))])
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title(f'Predicted Probabilities for Single Sample')
        plt.savefig('predicted_probabilities.png')
        tab_1.pyplot()
else :
    file = st.sidebar.file_uploader("Upload file to test")
    if file == None :
        st.write("Input File")
    else :
        df = pd.read_csv(file)
        tab_1.dataframe(df)
        X = pd.get_dummies(df)
        for i in X.columns :
            X[i] = X[i].astype('int8')

        if alg_option.lower() == 'logistic regression' :
          pred_lr = model_lr.predict(X)
          pred_probalr = model_lr.predict_proba(X)[:,1]*100
          df['pred_proba'] = pred_probalr
          df['predictions'] = pred_lr
          tab_2.write(df['predictions'].value_counts())
          tab_2.dataframe(df)

          fig = sns.countplot(data = df, x= "predictions")
          plt.savefig('predicted_probabilities.png')
          tab_2.pyplot()
        else :
            pred_dt = model_dt.predict(X)
            pred_probadt = model_dt.predict_proba(X)[:,1] * 100
            df['pred_proba'] = pred_probadt
            df["predictions"] = pred_dt
            tab_2.write(df['predictions'].value_counts())
            tab_2.dataframe(df)

            fig = sns.countplot(data = df, x= "predictions")
            plt.savefig('predicted_probabilities.png')
            tab_2.pyplot()
    @st.cache_data # <------------- IMPORTANT: Cache the conversion to prevent computation on every rerun

    def convert_df(df): # <--------------- Function declaration
        '''
        Convert_df function converts the resulting dataframe to a CSV file.
        It takes in a data frame as a aprameter.
        It returns a CSV file
        '''
        
        return df.to_csv().encode('utf-8') # <--------------- Return dataframe as a CSV file
    csv = convert_df(df) # <------------ Convert_df function calling and assigning to a variable.
    tab_2.success("Print Result as CSV file") # <--------------- A widget as heading for the download option in tab 2
    tab_2.download_button("Download",csv,"Prediction.csv",'text/csv') # <------------------ Download button widget in tab 2.


