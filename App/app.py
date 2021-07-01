# Importing necessary Packages

import streamlit as st
import altair as alt
import plotly.express as px
import pandas as pd
import numpy as np
import joblib

# Load the Model
pipeline_lr = joblib.load(open("../Model/emotion_classifier.pkl", "rb"))

# Functions
def predict_emotions(text):
    result = pipeline_lr.predict([text])
    return result[0]

def predict_probability(text):
    result = pipeline_lr.predict_proba([text])
    return result

st.title("Detectemo")
choice = st.sidebar.selectbox("Menu", ["Home", "About"])
if choice is "Home":
    st.subheader("Home - Emoticon in Text")
    with st.form("emoticon_clf_form"):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label = "Submit")
        
    if submit_text:
        col1, col2 = st.beta_columns(2)
        prediction = predict_emotions(raw_text)
        probability = predict_probability(raw_text)
        
        with col1:
            st.success("Original text")
            st.write(raw_text)
            st.success("Prediction")
            st.write(prediction)
            st.write("Confidence: ", np.max(probability))
            
        with col2:
            st.success("Prediction Probability")
            prob_df = pd.DataFrame(probability, columns = pipeline_lr.classes_)
            prob_df_clean = prob_df.T.reset_index()
            prob_df_clean.columns = ["emotions", "probability"]
            fig = alt.Chart(prob_df_clean).mark_bar().encode(x="emotions", y="probability", color="emotions")
            st.altair_chart(fig, use_container_width=True)
            
elif choice is "About":
    st.subheader("About")
