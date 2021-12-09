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

#getting the analysis data frame
clean_df = pd.read_csv('https://raw.githubusercontent.com/MafikengZ/NLP-Tweet-Sentiment-Ananlysis/main/API/resources/data/clean_df.csv')

def plot_target_based_features(feature):
    fig, ax = plt.subplots(1,1,'all', figsize = (15, 9))
    # create data frames for classes
    x1 = clean_df[clean_df['sentiment'] == 1][feature]
    x2 = clean_df[clean_df['sentiment'] == 2][feature]
    x0 = clean_df[clean_df['sentiment'] == 0][feature]
    x_neg = clean_df[clean_df['sentiment'] == -1][feature]
    #plt.figure(1, figsize = (16, 8))
    plt.xlabel('number of charecters in a tweet')
    plt.ylabel('number of tweets')
    plt.title('Word distribution chart')

    _ = plt.hist(x1, alpha = 0.5, color = 'grey', bins = 50, label = 'belivers')
    _ = plt.hist(x2, alpha = 0.5, color = 'blue', bins = 50, label = 'news')
    _ = plt.hist(x0, alpha = 0.6, color = 'green', bins = 50, label = 'neutral')
    _ = plt.hist(x_neg, alpha = 0.5, color = 'orange', bins = 50, label = 'anti')
    plt.legend(["belivers", 'news', 'neutral', 'anti'])

    st.pyplot(fig)

    return _

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

    st.title("Tweet Sentiment Classifier")
    st.subheader("There Is Only One Earth And One Life")
    st.write('let us make the right choice')

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Information", "Prediction"]
    selection = st.sidebar.selectbox("Choose Option", options)
    st.image("resources/imgs/worl.jpg",width=700)

    # Building out the "Information" page
    if selection == "Information":
        st.write('The war of good and evil between believers and non believers')

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
        class_options = ["Class Pro", "Class News", "Class Neutral", "Class Anti", "overall analysis"]
        class_selection = st.sidebar.selectbox("Choose Class Analysis", class_options)


        if class_selection == "Class Pro":
            #creating a page for the class pro
            st.write("This page helps the user with the analysis of the most trending tweets, the trending  hashtags "
                     "and trending topics in the classes in Class Pro")

            # Creating a bar chart for the 25 most occurring words of class 1
            class_1_data = df_token.sort_values('1', ascending = False).head(25)
            data_set1 = {
                'tokens': class_1_data['Unnamed: 0'],
                'values': class_1_data['1']
            }

            df1 = pd.DataFrame(data_set1).set_index('tokens')
            st.subheader("What Are The Trending Topics In Pro Class")
            st.write("The most trending topics in the pro Believers class seems to be about climate change  as expected")
            st.bar_chart(df1)
            st.write("The data shows the climate change as the most trending topic and also gives links of videos "
                     "and sites where people can read further about these topic this is seen fromm the urlweb token, "
                     "this would be a nature of someone who believes in something, they would want to share more.")

            st.subheader("Trending Tweets")
            st.write("The most trending tweets in the believers class seem to be about alerts and fights against this "
                     "global threat of global warming.")
            # Display the generated wordcloud for class 1:
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_pro, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            st.write('The believers do what anybody who believes in something would do, they talk about '
                     'believing and that is what we do most of the time, we talk about who we are, they talk '
                     'about tackling the threat, the talk about the scientists who talk about these things '
                     'they talk about their leaders, they talk about the enthusiast of Eco friendly environment.')

            # showing raw hashtags data
            if st.checkbox('See the most retweeted tweets '):
                st.subheader('Trending tweets and their counts')
                st.write(pro_retweet.value_counts())

            st.subheader("What Are The Trending Hashtags")
            st.write('The trending hashtags in the believers class are about taking the initiative of saving '
                     'the world, they are talking about the people and companies that are enthusiasts ')
            # Display the generated wordcloud for class 1:
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_pro1, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            st.write("projects like the one that has been initiated by the amp to mange and invest companies in"
                     "alignment with net zero emission by 2050.The EPA (environmental protection agency) which"
                     "deals with fighting the war of the global climate change")

            # showing raw hashtags data
            if st.checkbox('See trending hashtags'):
                st.subheader('Trending hashtags and their counts')
                st.write(pro_hash.value_counts())

        if class_selection == "Class News":
            st.write("This page helps the user with the analysis of the most trending tweets, the trending  hashtags "
                     "and trending topics in the classes in the Class News")

            # Creating a bar chart for the 25 most occurring words of class 2
            class_2_data = df_token.sort_values('2', ascending = False).head(25)

            data_set2 = {
                'tokens': class_2_data['Unnamed: 0'],
                'values': class_2_data['2']
            }
            st.subheader("What Are The Trending Topics in class News")
            st.write("""The most common topics in the news class are the climate and change
                        and we can see there are more url webs and news available.
                    """)
            df2 = pd.DataFrame(data_set2).set_index('tokens')
            st.bar_chart(df2)



            # Display the generated wordcloud for class 2:
            st.subheader("What Are Trending Tweets")
            st.write("""The most common things to the news people is send the the news about the trending 
            stories in the climate change topic.
                    """)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_news, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            st.write("""The most common things to the news people is send the the news about the trending 
            stories in the climate change topic. talking about Trump and EPA as the politics 
            on climate change, in the White House
                    """)

            # showing raw retweets data
            if st.checkbox('See trending tweets'):
                st.subheader('Trending tweets and their counts')
                st.write(news_retweet.value_counts())

            st.subheader("What Are The Trending Hashtags")
            st.write("The plans and talks around the US, India and Paris. The weather forecast from the "
                     "world. The China is known as the number one acting country in global warming. ")
            # Display the generated wordcloud for class 1:
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_news2, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            st.write("""The most common things to the news people is send the the news about the trending 
            stories in the climate change topic. talking about Trump and EPA as the politics 
            on climate change, in the White House
                    """)

            # showing raw hashtags data
            if st.checkbox('See trending hashtags'):
                st.subheader('Trending tweets and their counts')
                st.write(news_hash.value_counts())

        if class_selection == "overall analysis":
            st.write("This page helps the user with the analysis of all the classes combined to make a better "
                     "distribution of how data is distributed")
            # Create a bar chart for the column of sentiment value counts
            st.subheader("Sentiment Class count bar chart")
            st.write('The data distribution shows imbalance amongst the classes the positive class having'
                     ' the most data')
            data = raw['sentiment'].value_counts()
            st.bar_chart(data)

            # getting the lengths of tweets distribution
            st.subheader("What is the average length of Tweets from each class")
            st.write('Curiosity leads to asking what length of characters does each class use to voice out their '
                     'opinions in the fight of global warming, or their disbelief in it.')
            g1 = plot_target_based_features('length')

            st.subheader("What is the average number of words in a Tweet")
            st.write('Curiosity leads to asking how many words does each class use to voice out their '
                     'opinions in the fight of global warming, or their disbelief in it. with this information we can '
                     'strt thinking about how much each community shares, cause you only share when you '
                     'have something to share')
            # getting the length of number of words in a tweet
            g2 = plot_target_based_features('no_words')

            st.subheader("What is the average number of unique words in a Tweet")
            st.write('with the unique number of words we can try to find the vocabulary of these classes '
                     'but it seems as if the average number of unique words used by the believers are more, '
                     'interestingly followed by the Neutral the anti believers and then the news,'
                     'you would think the news would have more to say')
            #Getting the number of unique words distribution
            g3 = plot_target_based_features('unique_words')


            # getting the distribution of less common words




        if class_selection == "Class Neutral":


            # Creating a bar chart for the 25 most occurring words of class 0
            class_0_data = df_token.sort_values('0', ascending = False).head(25)
            data_set0 = {
                'tokens': class_0_data['Unnamed: 0'],
                'values': class_0_data['0']
            }
            st.subheader("What Are The Trending Topics In The Neutral Class")
            df0 = pd.DataFrame(data_set0).set_index('tokens')
            st.write("""The Neutral Tweets obviously ask and think a lot about the global climate change
                    """)
            st.bar_chart(df0)

            # Display the generated wordcloud for class 1:
            st.subheader("What AreTrending Tweets")
            st.write('The people who want to know always seek information and ask a lot of questions')
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_neutral, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            st.write("""The Neutral tweets show that people are kept in the dark about the things
            that also contribute to the climate, although they talk about the penguin ans the leaders,
            but they also want to know what to expect.
                    """)

            #show trending retweets
            if st.checkbox('Show trending tweets'):
                st.subheader('Center Information data')
                st.write(neutral_retweet.value_counts())

            st.subheader("Trending Hashtags")
            st.write("""The most common topics in hashtags are talking about the research done in various
            institutions, they ask a lot of questions expecting answers of course.
                    """)
            # Display the generated wordcloud for class 1:
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_neutral0, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            st.write("""The Tsinghua university of China is leading major investigations on climate 
            change. DiCaprio is an enthusiast of this good change of looking out for the planet.
                    """)

            # show raw trending hashtags
            if st.checkbox('Show trending hashtags'):
                st.subheader('Center Information data')
                st.write(neutral_hash.value_counts())

        if class_selection == "Class Anti":


            # Creating a bar chart for the 25 most occurring words of class -1
            class_neg_data = df_token.sort_values('-1', ascending = False).head(25)

            data_set_neg = {
                'tokens': class_neg_data['Unnamed: 0'],
                'values': class_neg_data['-1']
            }
            st.subheader("Top Trending topics")
            st.write("""To the anti believers think the whole thing is a scam, it is just a way to stop the US progress
            in production since the production stream requires a lot of dumping and, the emission of co2, thus
            they need to be st
                    """)
            df_neg = pd.DataFrame(data_set_neg).set_index('tokens')
            st.bar_chart(df_neg)


            # Display the generated wordcloud for class -1:
            st.subheader("Trending Tweets")
            st.write("""The concerns are raised in the tax contribution of the climate change initiation""")
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_anti, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            st.write("""The concerns are raised in the tax contribution of the climate change initiation. They think
            the global warming crisis is just a hoax from China. The Trump and Obama beef on their outlook
            on climate change seems to show.
                    """)


            if st.checkbox('Show trending tweets'):
                st.subheader('Trending tweets and their counts')
                st.write(anti_retweet.value_counts())

            st.subheader("Trending Hashtags")
            st.write("""The hashtags talks about being faked out, it also talks about the coal industries which use it 
            for produvtion.
                    """)
            # Display the generated wordcloud for class 1:
            fig, ax = plt.subplots()
            ax.imshow(wordcloud_antineg, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            st.write("""The hashtags also talk about the things that scientists have once said will happen
            but did not because of climate change, they talk about the normality of seasons based on what they
            usually receive in a season based on the weather conditions.
                    """)

            if st.checkbox('Show trending hashtags and counts'):
                st.subheader('Trending hashtags')
                st.write(anti_hash.value_counts())

            if st.checkbox('Show Rare words in the tweets'):
                st.subheader('Trending hashtags')
                st.write(df_token.head())


# Building out the predication page
    if selection == "Prediction":

        
        # choosing the algorithm
        algo_option = ["Linear_Logistics", "SVC_Linear", "Naive_Bayes"]
        algo_selection = st.sidebar.selectbox("Choose An Algorithm of choice", algo_option)


        st.info("Predicting with a " + algo_selection +  " Model")



        # Selecting a Linear Logistic regression
        if algo_selection == 'Linear_Logistics':
            pred = joblib.load(open(os.path.join("resources/Linear_Logistics_final_3.pkl"),"rb"))
            model_info = "The Logistics regression algorithm is a parametric equation algorithm, which " \
                         "uses regularization to penalise the feature significance and with the penalty choice " \
                         "of `L1` or `L2` we can use the regularization coeficient `C` to control the amount of penelisation" \
                         " we are giving to the features. Since the logistic regression algorithm is not a multi " \
                         "class algorithm, we use the `multi` option of the algorithm to create a classification," \
                         " for all the classes."
            st.write(model_info)


        # Creating the selection for the other algorithms
        if algo_selection == 'SVC_Linear':
            pred = joblib.load(open(os.path.join("resources/SVC_linear_final_4.pkl"),"rb"))
            model_info = "The SVC Linear kernel model utilises the `kernel` optimiser function to " \
                         "use a linear hyper plane to separate the classes or classify the " \
                         "classes. The plane is fitted in the middle of the classes, and where it " \
                         "touches on either sides of the classes, we call that the supports of the planes, " \
                         "this makes it less vulnerable to outliers, not unless the supports are moved"
            st.write(model_info)
                
        # Creating the selection for the other algorithms
        if algo_selection == 'Naive_Bayes':
            pred = joblib.load(open(os.path.join("resources/MultiNB_final_3.pkl"),"rb"))
            model_info = "The Naive Bayes Algorithm uses the probability of a feature being in a " \
                         "certain class by using the probability of the feature being in a class," \
                         " it is not a parametric algorithm," \
                         " thus it is not vulnerable to outliers " \
                         "but it is vulnerable to multicollinearity."
            st.write(model_info)

        # Adding algorithm description
        #st.write(model_info)
        # Creating a text box for user input
        tweet_text = st.text_area("Enter Text","Type Here ")
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
            # info about polarity and subjectivity
            st.markdown("The prediction Probability to help the analyst with the confidence"
                        " of predicting that particular class")


            with col1:
                probs = predictor.predict_proba(vect_text)



                # Getting the confidence
                st.success("Confidence : {}".format(np.max(probs)))
            with col2:

                # creating polarity and subjectivity list
                polarity_subjectivity = [TextBlob(tweet_text).sentiment[0], TextBlob(tweet_text).sentiment[1]]

                st.success("Text Categorized as: {}".format(output[prediction[0]]))

            # Making the graphs
            prob_df = pd.DataFrame(probs,  columns = predictor.classes_).T.reset_index()
            p_df = prob_df.set_index('index')

            st.bar_chart(p_df)
            st.write('')




# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
    main()
