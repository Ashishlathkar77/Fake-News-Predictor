import streamlit as st
import pandas as pd
import joblib

vectorizer = joblib.load('vectorizer.joblib')

model = joblib.load('model.joblib')

# page setting
st.set_page_config(page_title="Fake News Prediction System",  layout="centered")

# page header
st.title(f"Fake News Prediction System")

#features = [end_trackdone, start_trackdone, start_fwdbtn, end_fwdbtn, end_backbtn, no_pause_before_play, long_pause_before_play,
#            valence, vector_6, vector_5, duration, dyn_range_mean, vector_1, organism, energy, vector_2, us_popularity,
#            bounciness, short_pause_before_play, beat_strength]

with st.form("Prediction_form"):
    # form header
    st.header("Enter the specifications:")
    # input elements
    type_content = st.text_input(label="content: ")
    
    # submitt values
    submit_values = st.form_submit_button("Predict")

if submit_values:

    cols_to_scale = {'content':type_content}
    

    # create scaling input dataframe
    scaling_input = pd.DataFrame(type_content)

    scaling = vectorizer.transform(scaling_input)

    scaled = pd.DataFrame(scaling, columns=type_content)

    
    input_dict = {'content': type_content}
    
    # create input dataframe
    input_dataframe = pd.DataFrame(input_dict)

    
    # make predictions
    prediction = model.predict(input_dataframe)

    if prediction == 0:
        value = "Real News"
    elif prediction == 1:
        value = "Fake News"
    
    # output header
    st.header("Here is the prediction: ")
    # output results
    st.success(f'The user {value} the song')
    # balloons..
    st.balloons()