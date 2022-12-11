"""

    Simple Streamlit webserver application for serving developed classification
    models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
    application. You are expected to extend the functionality of this script
    as part of your predict project.

    For further help with the Streamlit framework, see:

    https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os
from PIL import Image
from streamlit_option_menu import option_menu
#from pathlib import path


# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/vectorizer_E.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
    """ ES2 Tech's Tweet Classifier App """
    
    # Creates a main title and subheader on your page
    # these are static across all pages
    st.title("ES2 Tech's Tweet Classifier")
    st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    with st.sidebar:
        selected = option_menu("Main Menu", ["Home", 'Information', "Prediction", "Contact Us"], 
        icons=['house', 'info-square', 'bar-chart-line', "envelope"], menu_icon="cast", default_index=1)
    selected
    #options = ["🏠Home", "ℹ️ Information", "📈Prediction", "☎️Contact Us" ]
    #selection = st.sidebar.selectbox("Select Option", options)
    if selected == "Home" :
        image1 = Image.open("resources/imgs/final_logo.png")
        st.image(image1)
        st.info(" ") 
    # Building out the "Information" page
    if selected == "Information":
        st.info("General Information")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")

        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            st.write(raw[['sentiment', 'message']]) # will write the df to the page

    # Building out the predication page
    if selected == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text","Type Here")

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = joblib.load(open(os.path.join("resources/SVC_linear_Final_ES2.pkl"),"rb"))
            prediction = predictor.predict(vect_text)

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(prediction))
            
        with st.expander("ℹ️ How to interpret the results", expanded=False):
            st.write(
            """
            Sentiment is categorized into 4 classes:\n
            [-1] = **Anti**: the tweet does not believe in man-made climate change \n
            [0] = **Neutral**: the tweet neither supports nor refutes the belief of man-made climate change \n
            [1] = **Pro**: the tweet supports the belief of man-made climate change \n
            [2] = **News**: the tweet links to factual news about climate change \n
        
            """
        )
        st.write("")
        
     # Building out the "Contact Us" page
    if selected == "Contact Us":
        image1 = Image.open("resources/imgs/final_logo.png")
        st.image(image1)
        st.markdown("For general enquiries or assistance with the app, our contact details are below:")
        st.markdown("☎️: 067 365 4200")
        st.markdown("📧: help@es2tech.com")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    main()
