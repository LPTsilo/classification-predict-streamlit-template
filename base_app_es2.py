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
from streamlit_metrics import metric, metric_row
#from pathlib import path


# Data dependencies
import pandas as pd


# Vectorizer
news_vectorizer = open("resources/vectorizer_E.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
raw2 = pd.read_csv("resources/train_data.csv")
# Separate sentiment into classes
sentiment_class = {-1: "Anti", 0: "Neutral", 1: "Pro", 2: "News"}
raw2["sentiment"] = raw2["sentiment"].apply(lambda num: sentiment_class[num])

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

    if selected == "Home" :
        image1 = Image.open("resources/imgs/logo_lebuso.jpg")
        image2 = Image.open("resources/imgs/Capture.PNG")
        st.image(image1)
        tab1, tab2 = st.tabs(["Who Are We?", "Our Team"])
        with tab1:

            st.markdown("""ES2 TECH is a market and AI research agency that specializes in convenient and cost-effective quantitative research studies across Africa and the rest of the world through artificial intelligence services.
            Our consultants are highly experienced data scientists, ML engineers, and AI consultants
            No matter your company’s size, data is an invaluable resource for making informed decisions.
            With us you can be sure that you are leveraging one of the most highly experienced agencies in the world. Our vast experience helps businesses like yours discover value from their data.
            Our data science consultants deliver incredible value by evaluating and recommending strategic business decisions to further your organizational ambitions""")

        with tab2:
            st.image(image2)



    # Building out the "Information" page
    if selected == "Information":
        st.markdown("")
        st.markdown("")
        st.markdown("")
        # You can read a markdown file from supporting resources folder
        st.markdown("Data insights")
        st.subheader("Raw Twitter data and label")
        if st.checkbox('Show raw data'): # data is hidden if box is unchecked
            st.write(raw[['message']]) # will write the df to the page
            st.subheader('Summary')
            metric_row(
                {
                    "% 😡 Negative Tweets": "{:.0%}".format(len(raw['sentiment'] == -1) / len(raw)),
                    "% 😑 Neutral Tweets": "{:.0%}".format(len(raw['sentiment'] == 0)/len(raw)),
                    "% 😃 Positive Tweets": "{:.0%}".format(len(raw['sentiment'] == 1)/len(raw))
                }
            )
        st.info("A Bar Graph showing the number of tweets per sentiment")
        st.bar_chart(data=raw2["sentiment"].value_counts(), x=None, y=None, width=220, height=320, use_container_width=True)
        st.info("As it is observed from the bar graph above, it is well noted that many people \port the belief of man-made climate change.")

            
    # Building out the predication page
    if selected == "Prediction":
        st.info("To test classifier accuracy, copy and past one of the tweets in the list below into the classifier and check the corresponding sentiment that the model outputs.")
        
        with st.expander("Tweets", expanded=False):
                st.write(
                """
                * The biggest threat to mankind is NOT global warming but liberal idiocy👊🏻🖕🏻\n
                Expected output = -1 \n
                * Polar bears for global warming. Fish for water pollution.\n
                Expected output = 0 \n
                * RT Leading the charge in the climate change fight - Portland Tribune  https://t.co/DZPzRkcVi2 \n
                Expected output = 1 \n
                * G20 to focus on climate change despite Trump’s resistance \n
                Expected output = 2
        
                """
            )
                st.write("")
        # Creating a text box for user input
        tweet_text = st.text_area(label="Enter Text", height= 100, help="Enter a text, then click on 'Classify' below", placeholder="Enter any text here")

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

        if selected == "Contact Us":
            with st.form("form1", clear_on_submit=True):
                name = st.text_input("Enter full name")
                email = st.text_input("Enter email")
                message = st.text_area("Message")

                submit = st.form_submit_button("Submit Form")

# Required to let Streamlit instantiate our web app.
if __name__ == '__main__':
    main()


