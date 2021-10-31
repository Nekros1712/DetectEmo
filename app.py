# Importing necessary Packages

import streamlit as st
import streamlit.components.v1 as components
import altair as alt
import plotly.express as px
import pandas as pd
import numpy as np
import joblib

# Load the Model
pipeline_lr = joblib.load(open("Model/LogisticRegression.pkl", "rb"))
pipeline_svc = joblib.load(open("Model/SupportVectorClassifier.pkl", "rb"))

# Functions
def predict_emotions(pipeline, text):
    result = pipeline.predict([text])
    return result[0]

def predict_probability(pipeline, text):
    result = pipeline.predict_proba([text])
    return result

st.title("Detectemo")
choice = st.sidebar.selectbox("Menu", ["Home", "About"])
if choice is "Home":
    st.subheader("Only thing that matters is Emotion")
    with st.form("emoticon_clf_form"):
        raw_text = st.text_input("Type Here")
        submit_text = st.form_submit_button(label = "Submit")
        
    if submit_text:
        prediction_lr = predict_emotions(pipeline_lr, raw_text)
        prediction_svc = predict_emotions(pipeline_svc, raw_text)
        probability_lr = predict_probability(pipeline_lr, raw_text)
        probability_svc = predict_probability(pipeline_svc, raw_text)
        
        col1, col2 = st.beta_columns(2)
        
        with col1:
            components.html(
                f'''
                    <div style="text-align: center; color: white; font-size: 25px;"> Logistic Regression </div>
                ''',
                height = 40
            )
            st.success("Prediction")
            st.write(prediction_lr)
            st.write("Confidence: ", np.max(probability_lr))
            st.success("Prediction Probability")
            prob_df = pd.DataFrame(probability_lr, columns = pipeline_lr.classes_)
            prob_df_clean = prob_df.T.reset_index()
            prob_df_clean.columns = ["emotions", "probability"]
            fig = alt.Chart(prob_df_clean).mark_bar().encode(x="emotions", y="probability", color="emotions")
            st.altair_chart(fig, use_container_width=True)
            
        with col2:
            components.html(
                f'''
                    <div style="text-align: center; color: white; font-size: 25px;"> Support Vector Classifier </div>
                ''',
                height = 40
            )
            st.success("Prediction")
            st.write(prediction_svc)
            st.write("Confidence: ", round(np.max(probability_svc), 4))
            st.success("Prediction Probability")
            prob_df = pd.DataFrame(probability_svc, columns = pipeline_svc.classes_)
            prob_df_clean = prob_df.T.reset_index()
            prob_df_clean.columns = ["emotions", "probability"]
            fig = alt.Chart(prob_df_clean).mark_bar().encode(x="emotions", y="probability", color="emotions")
            st.altair_chart(fig, use_container_width=True)
            
elif choice is "About":
    st.markdown("## About")
    st.write("DetectEmo: An Emotion Detector App\nType a sentence and this app will show you the emotion of it using two of the best models along with the confidence.")
    st.markdown('''## Developer: Samprit Chaurasiya''')
    st.markdown('''### [Github](https://github.com/Nekros1712) [Instagram](https://instagram.com/samprit1712)
    ''')
