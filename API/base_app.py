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
import numpy as np

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

# hash tags negative class
raw = pd.read_csv("resources/train.csv")

# Load Anti class data
anti_hash = pd.read_csv("https://raw.githubusercontent.com/MafikengZ/NLP-Tweet-Sentiment-Ananlysis/main/API/resources/data/df_anti_hashtags.csv")
anti_retweet = pd.read_csv("https://raw.githubusercontent.com/MafikengZ/NLP-Tweet-Sentiment-Ananlysis/main/API/resources/data/df_Anti_retweets.csv")

# Load Anti class data
neutral_hash = pd.read_csv("https://raw.githubusercontent.com/MafikengZ/NLP-Tweet-Sentiment-Ananlysis/main/API/resources/data/df_Neutral_hashtags.csv")
neutral_retweet = pd.read_csv("https://raw.githubusercontent.com/MafikengZ/NLP-Tweet-Sentiment-Ananlysis/main/API/resources/data/df_Neutral_retweet.csv")

# Load Anti class data
pro_hash = pd.read_csv("https://raw.githubusercontent.com/MafikengZ/NLP-Tweet-Sentiment-Ananlysis/main/API/resources/data/df_Pro_hashtags.csv")
pro_retweet = pd.read_csv("https://raw.githubusercontent.com/MafikengZ/NLP-Tweet-Sentiment-Ananlysis/main/API/resources/data/df_Pro_retweets.csv")

# Load Anti class data
news_hash = pd.read_csv("https://raw.githubusercontent.com/MafikengZ/NLP-Tweet-Sentiment-Ananlysis/main/API/resources/data/df_News_hashtags.csv")
news_retweet = pd.read_csv("https://raw.githubusercontent.com/MafikengZ/NLP-Tweet-Sentiment-Ananlysis/main/API/resources/data/df_News_retweets.csv")

# Load word count dataframe
token_df = pd.read_csv("https://raw.githubusercontent.com/MafikengZ/NLP-Tweet-Sentiment-Ananlysis/main/API/resources/data/df_token_ind.csv")



# Making sentences for word  cloud retweets
tweet_pro = " ".join([review for review in pro_retweet.Retweets if review is not np.nan])
tweet_neutral = " ".join(review for review in neutral_retweet.Retweets if review is not np.nan)
tweet_news = " ".join(review for review in news_retweet.Retweets if review is not np.nan)
tweet_anti = " ".join(review for review in anti_retweet.Retweets if review is not np.nan)

# Making sentences for word cloud hashtags
hash_pro = " ".join([review for review in pro_hash.hashtags if review is not np.nan])
hash_neutral = " ".join(review for review in neutral_hash.hashtags if review is not np.nan)
hash_news = " ".join(review for review in news_hash.hashtags if review is not np.nan)
hash_anti = " ".join(review for review in anti_hash.hashtags if review is not np.nan)


# The main function where we will build the actual app
def main():
    """Tweet Classifier App \with Streamlit """

    # Creates a main title and subheader on your page -
    # these are static across all pages

    st.title("Tweet Classifier")


    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Information", "Prediction"]
    selection = st.sidebar.selectbox("Choose Option", options)
    

        
    # Building out the "Information" page
    if selection == "Information":

        # Create and generate a word cloud image:
        wordcloud_pro = WordCloud(max_font_size=50, max_words=100,
                                  background_color="white").generate(tweet_pro)
        wordcloud_neutral = WordCloud(max_font_size=50, max_words=100
                                      , background_color="white").generate(tweet_neutral)
        wordcloud_news = WordCloud(max_font_size=50, max_words=100,
                                   background_color="white").generate(tweet_news)
        wordcloud_anti = WordCloud(max_font_size=50, max_words=100,
                                   background_color="white").generate(tweet_anti)

        # Create and generate a word cloud image:
        wordcloud_pro1 = WordCloud(max_font_size=50, max_words=100,
                                  background_color="white").generate(hash_pro)
        wordcloud_neutral0 = WordCloud(max_font_size=50, max_words=100
                                      , background_color="white").generate(hash_neutral)
        wordcloud_news2 = WordCloud(max_font_size=50, max_words=100,
                                   background_color="white").generate(hash_news)
        wordcloud_antineg = WordCloud(max_font_size=50, max_words=100,
                                   background_color="white").generate(hash_anti)

        #copy of the data frame
        df_token = token_df.copy()



        #making the class analysis
        class_options = ["Class Pro", "Class News", "Class Neutral", "Class Anti"]
        class_selection = st.sidebar.selectbox("Choose Class Analysis", class_options)


        if class_selection == "Class Pro":
            # Create a bar chart for the column of sentiment value counts
            st.subheader("Sentiment Class distribution")
            data = raw['sentiment'].value_counts()
            st.bar_chart(data)

            # Creating a bar chart for the 25 most occurring words of class 1
            class_1_data = df_token.sort_values('1', ascending = False).head(25)
            data_set1 = {
                'tokens': class_1_data['Unnamed: 0'],
                'values': class_1_data['1']
            }

            df1 = pd.DataFrame(data_set1).set_index('tokens')
            st.subheader("25 most common tokens in Class Pro")
            st.bar_chart(df1)

            st.subheader("Trending Tweets")
            # Display the generated wordcloud for class 1:
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_pro, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

            # showing raw hashtags data
            if st.checkbox('Show trending tweets'):
                st.subheader('Trending tweets and their counts')
                st.write(pro_retweet.value_counts())

            st.subheader("Trending Hashtags")
            # Display the generated wordcloud for class 1:
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_pro1, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

            # showing raw hashtags data
            if st.checkbox('Show trending hashtags'):
                st.subheader('Trending tweets and their counts')
                st.write(pro_hash.value_counts())

        if class_selection == "Class News":
            # Create a bar chart for the column of sentiment value counts
            st.subheader("Sentiment Class count bar chart")
            data = raw['sentiment'].value_counts()
            st.bar_chart(data)


            # Creating a bar chart for the 25 most occurring words of class 2
            class_2_data = df_token.sort_values('2', ascending = False).head(25)

            data_set2 = {
                'tokens': class_2_data['Unnamed: 0'],
                'values': class_2_data['2']
            }
            st.subheader("25 Most common tokens in class News")
            df2 = pd.DataFrame(data_set2).set_index('tokens')
            st.bar_chart(df2)


            # Display the generated wordcloud for class 2:
            st.subheader("Trending Tweets")
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_news, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

            # showing raw retweets data
            if st.checkbox('Show trending tweets'):
                st.subheader('Trending tweets and their counts')
                st.write(news_retweet.value_counts())

            st.subheader("Trending Hashtags")
            # Display the generated wordcloud for class 1:
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_news2, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

            # showing raw hashtags data
            if st.checkbox('Show trending hashtags'):
                st.subheader('Trending tweets and their counts')
                st.write(news_hash.value_counts())

        if class_selection == "Class Neutral":
            # Create a bar chart for the column of sentiment value counts
            st.subheader("Sentiment Class count bar chart")
            data = raw['sentiment'].value_counts()
            st.bar_chart(data)


            # Creating a bar chart for the 25 most occurring words of class 0
            class_0_data = df_token.sort_values('0', ascending = False).head(25)
            data_set0 = {
                'tokens': class_0_data['Unnamed: 0'],
                'values': class_0_data['0']
            }
            st.subheader("25 most common Tokens")
            df0 = pd.DataFrame(data_set0).set_index('tokens')
            st.bar_chart(df0)


            # Display the generated wordcloud for class 1:
            st.subheader("Trending Tweets")
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_neutral, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

            #show trending retweets
            if st.checkbox('Show trending tweets'):
                st.subheader('Center Information data')
                st.write(neutral_retweet.value_counts())

            st.subheader("Trending Hashtags")
            # Display the generated wordcloud for class 1:
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_neutral0, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

            # show raw trending hashtags
            if st.checkbox('Show trending hashtags'):
                st.subheader('Center Information data')
                st.write(neutral_hash.value_counts())

        if class_selection == "Class Anti":
            # Create a bar chart for the column of sentiment value counts
            st.subheader("Sentiment Class count bar chart")
            data = raw['sentiment'].value_counts()
            st.bar_chart(data)


            # Creating a bar chart for the 25 most occurring words of class -1
            class_neg_data = df_token.sort_values('-1', ascending = False).head(25)

            data_set_neg = {
                'tokens': class_neg_data['Unnamed: 0'],
                'values': class_neg_data['-1']
            }
            st.subheader("25 most common Tokens")
            df_neg = pd.DataFrame(data_set_neg).set_index('tokens')
            st.bar_chart(df_neg)


            # Display the generated wordcloud for class -1:
            st.subheader("Trending Tweets")
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_anti, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)


            if st.checkbox('Show trending tweets'):
                st.subheader('Trending tweets and their counts')
                st.write(anti_retweet.value_counts())

            st.subheader("Trending Hashtags")
            # Display the generated wordcloud for class 1:
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_antineg, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

            if st.checkbox('Show trending hashtags and counts'):
                st.subheader('Trending hashtags')
                st.write(anti_hash.value_counts())

            if st.checkbox('Show Rare words in the tweets'):
                st.subheader('Trending hashtags')
                st.write(df_token.head())


# Building out the predication page
    if selection == "Prediction":

        
        # choosing the algorithm
        algo_option = ["Linear_Logistics", "SVM_Linear", "Naive_Bayes"]
        algo_selection = st.sidebar.selectbox("Choose An Algorithm of choice", algo_option)


        st.info("Predicting with a " + algo_selection +  " Model")
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text","Type Here ")


        # Selecting a Linear Logistic regression
        if algo_selection == 'Linear_Logistics':
            pred = joblib.load(open(os.path.join("resources/Linear_Logistics_final_3.pkl"),"rb"))

        # Creating the selection for the other algorithms
        if algo_selection == 'SVM_Linear':
            pred = joblib.load(open(os.path.join("resources/SVC_linear_final_4.pkl"),"rb"))
                
        # Creating the selection for the other algorithms
        if algo_selection == 'Naive_Bayes':
            pred = joblib.load(open(os.path.join("resources/MultiNB_final_3.pkl"),"rb"))


        if st.button("Classify"):
            col1, col2 = st.columns(2)
            # Transforming user input with vectorizer
            vect_text = tweet_cv.transform([tweet_text]).toarray()
            # Load your .pkl file with the model of your choice + make predictions
            # Try loading in multiple models to give the user a choice
            predictor = pred
            prediction = predictor.predict(vect_text)

            output = {0 : 'Neutral tweet', 1 : 'Pro Believer',
                      2 : 'News facts', -1 : 'Anti believer'}


                    

            # When model has successfully run, will print prediction
            # You can use a dictionary or similar structure to make this output
            # more human interpretable.


            with col1:
                probs = predictor.predict_proba(vect_text)

                # info about polarity and subjectivity
                st.markdown("The prediction Probability to help the analyst with the confidence"
                            " of predicting that particular class")

                # Getting the confidence
                st.success("Confidence : {}".format(np.max(probs)))

                # Making the graphs
                prob_df = pd.DataFrame(probs,  columns = predictor.classes_).T.reset_index()
                p_df = prob_df.set_index('index')

                st.bar_chart(p_df)



            with col2:

                # creating polarity and subjectivity list
                polarity_subjectivity = [TextBlob(tweet_text).sentiment[0], TextBlob(tweet_text).sentiment[1]]

                # info about polarity and subjectivity
                st.markdown("The Polarity helps tell if tweet is negative or positive "
                            "subjectivity defining if fact or opinion")

                st.success("Text Categorized as: {}".format(output[prediction[0]]))

                # Making the graphs
                data_set = {
                    'measures': ['Polarity', 'Subjectivity'],
                    'values': polarity_subjectivity
                }

                df = pd.DataFrame(data_set)

                st.bar_chart(df['values'])




# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    main()
