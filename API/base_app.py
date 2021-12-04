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

# Data plots
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import altair as alt

# data manipulation
import string
import regex as re

# word tokenize
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
import nltk
from nltk.corpus import stopwords


# Vectorizer
news_vectorizer = open("resources/vectorizer_3.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")



# The main function where we will build the actual app
def main():
    """Tweet Classifier App \with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages

    st.title("Tweet Classifier")

    st.subheader("Climate change tweet classification")

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    

        
    # Building out the "Information" page
    if selection == "Information":



        st.markdown("""
                        | Sentiment     | Meaning       | 
                        | ------------- |:-------------:| 
                        | 1             | Pro believers |
                        | 2             | News facts    |   
                        | 0             | Neutral tweet |
                        | -1            | Non believers  |    """)

        # You can read a markdown file from supporting resources folder
        st.markdown("Since the data is Imbalanced the use of sampling was the help for making good"
                    " predictions for other classes. The Pro Class seems to be advantaged since it has more"
                    " data points in the training of the model")

        # Create two columns one for bar chart and one for raw data
        # Create a bar chart for the column of sentiment value counts                
        st.subheader("Sentiment count bar chart")
        data = raw['sentiment'].value_counts()
        st.bar_chart(data)

        # Create a second column with raw data
        st.subheader("Raw data")
        #st.write(raw[['sentiment', 'message']]) # will write the df to the


        # lowering tokens
        raw['message'] = raw['message'].str.lower()



        # Removing Emojis
        emoji_pattern = re.compile("["
                                       u"\U0001F600-\U0001F64F"  # emoticons
                                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                       u"\U00002702-\U000027B0"
                                       u"\U000024C2-\U0001F251"
                                       u"\U0001f926-\U0001f937"
                                       u'\U00010000-\U0010ffff'
                                       u"\u200d"
                                       u"\u2640-\u2642"
                                       u"\u2600-\u2B55"
                                       u"\u23cf"
                                       u"\u23e9"
                                       u"\u231a"
                                       u"\u3030"
                                       u"\ufe0f"
                                       "]+", flags=re.UNICODE)

        x = [emoji_pattern.sub(r'', string) for string in raw['message']]

        raw['no_emoji'] = x

        #remmoving the emojis
        pattern_url = r'http[s]?://[A-Za-z0-9/.]+'
        subs_url = r'url-web'
        raw['clean_messages'] = raw['no_emoji'].replace(to_replace = pattern_url, value = subs_url, regex = True)

        #remmoving the uknown charecters from words
        pattern_url = r'[^A-Za-z ]'
        subs_url = r''
        raw['clean_messages'] = raw['clean_messages'].replace(to_replace = pattern_url, value = subs_url, regex = True)

        #remmoving the digits
        pattern_url = r'\d'
        subs_url = r''
        raw['clean_messages'] = raw['clean_messages'].replace(to_replace = pattern_url, value = subs_url, regex = True)


        #remmoving the Re Tweets
        pattern_url = r'rt\s'
        subs_url = r''
        raw['clean_messages'] = raw['clean_messages'].replace(to_replace = pattern_url, value = subs_url, regex = True)


        # tokenizing the tweets
        tokeniser = TreebankWordTokenizer()
        raw['tokens'] = raw['clean_messages'].apply(tokeniser.tokenize)

        st.write(raw)



    # Building out the predication page
    if selection == "Prediction":

        
        # choosing the algorithm
        algo_option = ["Linear_Logistics", "SVM_Linear", "Naive_Bayes"]
        algo_selection = st.sidebar.selectbox("Choose An Algorithm of choice", algo_option)



        st.info("Predicting with a " + algo_selection +  " Model")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text","Type Here")


        # Selecting a Linear Logistic regression
        if algo_selection == 'Linear_Logistics':
                pred = joblib.load(open(os.path.join("resources/Linear_Logistics_final_3.pkl"),"rb"))

        # Creating the selection for the other algorithms
        if algo_selection == 'SVM_Linear':
                pred = joblib.load(open(os.path.join("resources/SVC_linear_final_3.pkl"),"rb"))
                
        # Creating the selection for the other algorithms
        if algo_selection == 'Naive_Bayes':
                pred = joblib.load(open(os.path.join("resources/MultiNB_final_3.pkl"),"rb"))

        
 
                                                        

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = pred
            prediction = predictor.predict(vect_text)

            output = {0 : 'Neutral tweet about climate change', 1 : 'Pro Believer in climate change',
                      2 : 'News facts about climate change', -1 : 'Anti believer in climate change'}
                    

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.
            st.success("Text Categorized as: {}".format(output[prediction[0]]))

            # creating polarity and subjectivity list
            polarity_subjectivity = [TextBlob(tweet_text).sentiment[0], TextBlob(tweet_text).sentiment[1]]


            #
            st.markdown("The Polarity and subjectivity plots helps verify the sentiment by either "
                        "being a negative sentiment if the graph is negative and the subjectivity to help"
                        " predict how factual ones statement is, sometimes it cannot get sentiments of "
                        "words if there is word that helps with defining the tensity of the situation")

            # Making the graphs
            data_set = {
                'countries': ['Polarity', 'Subjectivity'],
                'values': polarity_subjectivity
            }

            df = pd.DataFrame(data_set)

            line = alt.Chart(df).mark_bar(color="Blue").encode(
                x='countries',
                y='values'
            ).properties(width=650, height=400, title="Polarity and Subjectivity plot").interactive()

            st.altair_chart(line)

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    main()
