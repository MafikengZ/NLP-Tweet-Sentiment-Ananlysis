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

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

image = Image.open("C:/Users/Mpilenhle/Documents/EDSA/Classification/classification-predict-streamlit-model/resources/imgs/worl.jpg")

# The main function where we will build the actual app
def main():
    """Tweet Classifier App \with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages

    st.title("Tweet Classifer")

    st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    

        
    # Building out the "Information" page
    if selection == "Information":

        st.image(image, caption='The futer of and now', width = 650)

        st.info("The Values of the Labes: 1 : Believes in Climate,\n 0 : Neutral about climate change, \n 2 : News facts about climate,\n -1 :  does not believe in climate change")
        # You can read a markdown file from supporting resources folder
        st.markdown("Some information here")

        # Create two columns one for bar chart and one for raw data
        # Create a bar chart for the column of sentiment value counts                
        st.subheader("Sentiment count bar chart")
        data = raw['sentiment'].value_counts()
        st.bar_chart(data)

        # Create a second column with raw data
        st.subheader("Raw data")
        st.write(raw[['sentiment', 'message']]) # will write the df to the page

    # Building out the predication page
    if selection == "Prediction":

        
        # choosing the algorithm
        algo_option = ["Linear_Logistics", "SVM_Linear", "Naive_Bayes", "SVM_poly", "Logistics_ovr" ]
        algo_selection = st.sidebar.selectbox("Choose An Algorith of choice", algo_option)

        st.markdown("""
                        | Sentiment     | Meaning       | 
                        | ------------- |:-------------:| 
                        | 1             | Pro believers |
                        | 2             | News facts    |   
                        | 0             | Neutral tweet |
                        | -1            | Non belivers  |    """)

        st.info("Predicting with a " + algo_selection +  " Model")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text","Type Here")

        
        # Selecting a Linear Logistic regression
        if algo_selection == 'Linear_Logistics':
                pred = joblib.load(open(os.path.join("C:/Users/Mpilenhle/Documents/EDSA/Classification/classification-predict-streamlit-model/resources/Logistic_regression.pkl"),"rb"))

        # Creating the selection for the other algorithms
        if algo_selection == 'SVM_Linear':
                pred = joblib.load(open(os.path.join("C:/Users/Mpilenhle/Documents/EDSA/Classification/classification-predict-streamlit-model/resources/SVC_linear.pkl"),"rb"))
                
        # Creating the selection for the other algorithms
        if algo_selection == 'Naive_Bayes':
                pred = joblib.load(open(os.path.join("C:/Users/Mpilenhle/Documents/EDSA/Classification/classification-predict-streamlit-model/resources/guassian.pkl"),"rb"))

        # Creating the selection for the other algorithms
        if algo_selection == 'Logistics_ovr':
                pred = joblib.load(open(os.path.join("C:/Users/Mpilenhle/Documents/EDSA/Classification/classification-predict-streamlit-model/resources/Logistics_ovr.pkl"),"rb"))

        # Creating the selection for the other algorithms
        if algo_selection == 'SVM_poly':
                pred = joblib.load(open(os.path.join("C:/Users/Mpilenhle/Documents/EDSA/Classification/classification-predict-streamlit-model/resources/SVC_poly.pkl"),"rb"))


                                                        

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = pred
            prediction = predictor.predict(vect_text)

            output = {0 : 'Neutral tweet', 1 : 'Pro Believer', 2 : 'News facts', -1 : 'Anti believer'}
                    

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(output[prediction[0]]))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    main()
