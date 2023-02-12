import tweepy as tw
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import nltk
from nltk.corpus import stopwords
from textblob import Word, TextBlob

consumer_key = 'xgptrNLUvzOM1XvmsMB2qGS3m'
consumer_secret = 'Ug50lFKeOL85LmMJDjsdQ5DVPMS2v7n0wsNEMcIDlxXsvF7Z17'
access_token = '1620894277294333960-tuEKFZ1KupMDHin5JpZR0RZJPmCuAr'
access_token_secret = 'aZ6BSkLqVCgWxZgy0RibtB2dWc5ecBK4FUYp2Je5Q25U0'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

hashtag = "#ChatGPT"
query = tw.Cursor(api.search, q=hashtag).items(1000)
tweets = [{'Tweet':tweet.text, 'Timestamp':tweet.created_at} for tweet in query]

df = pd.DataFrame.from_dict(tweets)

microsoft_handle = ['Microsoft', 'ChatGPT', 'Microsoft\'s']
google_handle = ['Google', 'GoogleAI', 'Google\'s']

def identify_subject(tweet, refs):
    flag = 0 
    for ref in refs:
        if tweet.find(ref) != -1:
            flag = 1
    return flag

df['Microsoft'] = df['Tweet'].apply(lambda x: identify_subject(x, microsoft_handle)) 
df['Google'] = df['Tweet'].apply(lambda x: identify_subject(x, google_handle))

stop_words = stopwords.words('english')
custom_stopwords = ['RT', '#ChatGPT']

def preprocess_tweets(tweet, custom_stopwords):
    processed_tweet = tweet
    processed_tweet.replace('[^\w\s]', '')
    processed_tweet = " ".join(word for word in processed_tweet.split() if word not in stop_words)
    processed_tweet = " ".join(word for word in processed_tweet.split() if word not in custom_stopwords)
    processed_tweet = " ".join(Word(word).lemmatize() for word in processed_tweet.split())
    return(processed_tweet)

df['Processed Tweet'] = df['Tweet'].apply(lambda x: preprocess_tweets(x, custom_stopwords))

df['polarity'] = df['Processed Tweet'].apply(lambda x: TextBlob(x).sentiment[0])
df['subjectivity'] = df['Processed Tweet'].apply(lambda x: TextBlob(x).sentiment[1])
df[['Processed Tweet', 'Microsoft', 'Google', 'polarity', 'subjectivity']].head()

df[df['Microsoft']==1][['Microsoft','polarity','subjectivity']].groupby('Microsoft').agg([np.mean, np.max, np.min, np.median])
df[df['Google']==1][['Google','polarity','subjectivity']].groupby('Google').agg([np.mean, np.max, np.min, np.median])

google = df[df['Google']==1][['Timestamp', 'polarity']]
google = google.sort_values(by='Timestamp', ascending=True)
google['MA Polarity'] = google.polarity.rolling(10, min_periods=3).mean()

microsoft = df[df['Microsoft']==1][['Timestamp', 'polarity']]
microsoft = microsoft.sort_values(by='Timestamp', ascending=True)
microsoft['MA Polarity'] = microsoft.polarity.rolling(10, min_periods=3).mean()

repub = 'red'
demo = 'blue'
fig, axes = plt.subplots(2, 1, figsize=(13, 10))

axes[0].plot(google['Timestamp'], google['MA Polarity'])
axes[0].set_title("\n".join(["Google Polarity"]))
axes[1].plot(microsoft['Timestamp'], microsoft['MA Polarity'], color='red')
axes[1].set_title("\n".join(["Microsoft Polarity"]))
fig.suptitle("\n".join(["ChatGPT Analysis"]), y=0.98)
plt.show()