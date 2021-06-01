# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 12:50:54 2021

@author: sneha
"""
# importing libraries
import tweepy #for twitter API
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import re
import string 
from langdetect import detect
from IPython.display import display


# assigning keys too 

twitterConsumerKey = "Enter your API key"
twitterConsumerSecretKey = "API secret key"
twitterAccessToken = "Acces Token"
twitterAccessTokenSecretKey = "Access token secret key"


# authentication of the account and keys

auth = tweepy.OAuthHandler(twitterConsumerKey,twitterConsumerSecretKey)
auth.set_access_token(twitterAccessToken,twitterAccessTokenSecretKey)
api = tweepy.API(auth)


# for analyzing the sentiments we need to calculate the positive, negative, neutral and compound parameters

# defining a function to calculate the percentage of sentiment parameters
def percent(part,whole):
    return 100* float(part)/ float(whole)


# we will limit the number of tweets that we need

word = input(" Enter topic of your interest : ")
ntweets = int(input(" Enter number of tweets to perform analysis on :"))
tweets = tweepy.Cursor(api.search, q=word,lang='en').items(ntweets)

# appending tweets in a list
list = []
for tweet in tweets:
    list.append(tweet.text)
 
    
# converting the list into pandas dataframe
df = pd.DataFrame(list)
df.columns=['Tweet']
# deleting duplicate columns
df.drop_duplicates(inplace = True)
    


# since the data consists of mentions(hastags), retweets we need to clean it
def cleaning(text):
    text = re.sub(r'@[A-Za-z0–9_]+','', text)
    #text = re.sub(r'#','', text)
    text = re.sub(r'RT:','', text)
    text = re.sub(r'https?:\/\/[A-Za-z0–9\.\/]+','', text)
    text = text.replace("'", "")
    text = text.replace('"', "")
    text = text.replace('`', "")
    return text.lower()
    
# applying cleaning function to the tweet data

df['Tweet'] = df['Tweet'].apply(cleaning)


# fidning opinion
def textSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# finding out emotions 
def textPolarity(text):
    return TextBlob(text).sentiment.polarity


df['Subjectivity'] = df['Tweet'].apply(textSubjectivity)
df['Polarity'] = df['Tweet'].apply(textPolarity)

# deleting entries with no data

df = df.drop(df[df['Tweet'] == ''].index)
df.head()

# classifying tweets by sentiments, considering polarity 
def textAnalysis(t):
    if t<0:
        return "Negative"
    elif t == 0:
        return "Neutral"
    else:
        return "Positive"
    
# creating a new column score for storing the sentiments
df['Score'] =df['Polarity'].apply(textAnalysis)

# calculating what percentage of tweets belong to each sentiment
neutral =  df[df['Score'] == "Neutral"]
print(str(neutral.shape[0]/(df.shape[0])*100) + "% of neutral tweets")
neutral_percent = neutral.shape[0]/(df.shape[0])*100
positive = df[df['Score'] == "Positive"]
print(str(positive.shape[0]/(df.shape[0])*100) + "% of positive tweets")
positive_percent = positive.shape[0]/(df.shape[0])*100
negative = df[df['Score'] == "Negative"]
print(str(negative.shape[0]/(df.shape[0])*100) + "% of negative tweets")
negative_percent = negative.shape[0]/(df.shape[0])*100


#Creating PieCart
explode=(0,0.1,0)
labels = 'Positive', 'Negative', 'Neutral'
sizes = [positive_percent, negative_percent, neutral_percent]
colors = ['yellowgreen', 'red','lightblue']
plt.pie(sizes,explode = explode, colors=colors,autopct='%1.1f%%', startangle=120)
plt.legend(labels, loc = (-0.05,0.05),shadow = True)
#plt.title(“Sentiment Analysis Result for keyword= "+keyword+""" )
plt.axis('equal')
plt.show()


# creating and displaying wordcloud for words in tweet
def wordcloud(txt):
    stopwords = set(STOPWORDS)
    w = WordCloud(background_color='white',max_words=3000,stopwords=stopwords,
                   repeat=True)
    w.generate(str(txt))
    w.to_file('w.jpg')
    path='w.jpg'
    display(Image.open(path))
    

tw_list = pd.DataFrame(list)
tw_list.columns=['Tweet']

wordcloud(positive['Tweet'].values)




#Calculating tweet’s lenght and word count
df['text_len'] = df['Tweet'].astype(str).apply(len)
df['text_word_count'] = df['Tweet'].apply(lambda x: len(str(x).split()))

round(pd.DataFrame(df.groupby("Score").text_len.mean()),2)
round(pd.DataFrame(df.groupby('Score').text_word_count.mean()),2)



# getting rid of stem words
# removing punctuations
def remove(txt):
     txt = ''.join([char for char in txt if char not in string.punctuation])
     txt = re.sub('[0-9]+', '',txt)
     return txt

# remove puncuations in the tweets
df['punc'] = df['Tweet'].apply(lambda x : remove(x))
df.head()
    
# tokenizaton 
def token(txt):
    txt = re.split('\W',txt)
    return txt

df['tk'] = df['punc'].apply(lambda x : token(x.lower()))

# getting rid of stop words
sw = nltk.corpus.stopwords.words('english')

def stopword(txt):
    txt= [ l for l in txt if l not in sw]
    return txt

df['stp'] = df['tk'].apply(lambda x: stopword(x))

#Appliyng Stemmer, helps to get rid of all stem words and consider only 
b = nltk.PorterStemmer()

def stm(txt):
    txt = [b.stem(l) for l in txt]
    return txt

df['stmd'] = df['stp'].apply(lambda x: stm(x))


# cleaning tweets and removing stem words
def cln(txt):
    cl = "".join([word.lower() for word in  txt if word not in 
                  string.punctuation])
    cl1 = re.sub('[0-9]+', '',cl)
    tks = re.split('\W', cl1)
    txt = [b.stem(word) for word in tks if word not in sw]
    return txt


df.head()




from sklearn.feature_extraction.text import TfidfVectorizer
vc = CountVectorizer(analyzer = cln)

cntVec = vc.fit_transform(df['Tweet'])
cntv = pd.DataFrame(cntVec.toarray(),columns= vc.get_feature_names())
cntv.head()

cnt = pd.DataFrame(cntv.sum())
cnt1 = cnt.sort_values(0,ascending= False).head(20)
cnt1[1:11]

def n_gram(corpus,range,n = None):
    v = CountVectorizer(ngram_range= range, stop_words = 'english').fit(corpus)
    bfg = v.transform(corpus)
    sm = bfg.sum(axis = 0 )
    freq = [(word, sm[0,i]) for word, i in v.vocabulary_.items()]
    freq =sorted(freq, key = lambda x: x[1], reverse=True)
    return freq[:n]


# getting bigram of the tweets
n2 = n_gram(df['Tweet'],(2,2),25)
n2

#getting trigram of the tweets

n3= n_gram(df['Tweet'],(3,3),25)
n3