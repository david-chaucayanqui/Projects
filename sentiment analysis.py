#Big Data in Finance
#prof. Thomas Renault
#David Chaucayanqui
#M2 Empirical Finance
#20-02-2019

###############################PROJECT1: STOCKTWITS & SENTIMENT ANALYSIS

########OBJECTIVE1
#1/ Choose one stock
#2/ Extract at least 10.000 messages about this stock on StockTwits
#3/ Store messages on a MongoDB database

import urllib.request
import json
import pymongo
import time
import TextBlob
from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client["ProjectFinal"]
id_list=[] 
cursor=db.stocktwits.find(()) 
for element in cursor:
    id_list.append(element["id"])
    
print(id_list)
print(min(id_list))
            
while True:
    if len(id_list)==0:
        url="https://api.stocktwits.com/api/2/streams/symbol/SNAP.json"
    else:
        min_id=str(min(id_list))
        url="https://api.stocktwits.com/api/2/streams/symbol/SNAP.json?max=" + min_id
        
        print(url)
        
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    
    for element in data["messages"]:
        id_list.append(element["id"]) 
        
        try:
            sentiment = element["entities"]["sentiment"]["basic"]
        except:
            sentiment = 0
            
        insert_element = {"id": element["id"], "created_at": element["created_at"], "user": element["user"]["username"], 
                          "body": element["body"], "sentiment": sentiment}
        try:
            result = db.stocktwits.insert_one(insert_element)
        except:
            pass

##time.sleep(18)-->to get data non stop every 18 seconds
            

#4/Find the top 10 words mostly used in bullish messages (positive) 
#and the top 10 words mostly used in bearish messages (negative)
            
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vectorizer=CountVectorizer(min_df=100)
cursor = db.stocktwits.find({})
corpusbull = []
corpusbear = []

for element in cursor:
    if element["sentiment"] =="Bullish":
            corpusbull.append(element["body"])
           
xbull = vectorizer.fit_transform(corpusbull)
print(xbull)
ybull = pd.DataFrame(xbull.A, columns=vectorizer.get_feature_names())
ybull.to_csv("bullish.csv")

cursor = db.stocktwits.find({})
for element in cursor:
    if element["sentiment"] =="Bearish":
            corpusbear.append(element["body"])

xbear = vectorizer.fit_transform(corpusbear)
print(xbear)
ybear = pd.DataFrame(xbear.A, columns=vectorizer.get_feature_names())
ybear.to_csv("bearish.csv")  

#stopwords (first limit of our code: we could have got ride of stopwords)
#bagofword (we could have created a graphic representation of Bullish and Bearish words)
            
sumbull=ybull.sum(axis = 0, skipna = True) 
sumbear=ybear.sum(axis = 0, skipna = True) 

sumbull1=sumbull.sort_values(axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last')
sumbear1=sumbear.sort_values(axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last')



########OBJECTIVE2
#1/Construct a quantitative sentiment indicator for one stock (daily frequency)

import os
import pandas as pd
import csv
import re

#we are using the professor Renault's lexicon
os.chdir('/Users/davidchaucayanqui/Desktop/Big Data in Finance/')
df = pd.read_csv('/Users/davidchaucayanqui/Desktop/Big Data in Finance/l1_lexicon.csv',sep = ';',encoding = 'latin1')

positive=[]
negative=[]

#make loop:
dfneg = df[df.sw < 0]
dfpos = df[df.sw > 0]
negative = dfneg["keyword"]
positive = dfpos["keyword"]
negative_list = negative.tolist()
positive_list = positive.tolist()

cursor = db.stocktwits.find({})
for element in cursor:
    #getting rid of punctuation marks
    splitsentencde=re.findall(r"[\w']+|[.,!?;$]", str(element["body"]))
    
    sentimentnum = 0
    for word in splitsentencde:
        if word in negative_list:
            sentimentnum = sentimentnum - 1
        if word in positive_list:
            sentimentnum = sentimentnum + 1

    s =  re.sub(r'[^0-9a-zA-Z]+',' ',str(element["body"]))
    print(element["id"])
    db.stocktwits.update_one({"id": element["id"]},{'$set':{'sentimentnum': sentimentnum, "ctext": s }}, upsert=False)
    
    print(s, sentimentnum)
    
cursor = db.stocktwits.find({},{"created_at":1, "sentimentnum":1,"body":1 })
df = pd.DataFrame(list(cursor))
print(df)

df["created_at"] = pd.to_datetime(df["created_at"],format='%Y/%m/%d')
df = df.set_index(pd.DatetimeIndex(df["created_at"]))
df=df.drop('created_at',1)
df=df.dropna()
#dailyfrequency
df = df.resample("D").mean()
print(df)


#2/ Extract stock prices from Yahoo! Finance 
os.chdir('/Users/davidchaucayanqui/Desktop/')
prices = pd.read_csv('SNAP.csv',sep = ',',encoding = 'latin1')

prices["Date"] = pd.to_datetime(prices["Date"],format='%Y/%m/%d')
prices = prices.set_index(pd.DatetimeIndex(prices["Date"]))
prices=prices.drop('Date',1)
prices=prices['Adj Close']

ret=prices / prices.shift(1) - 1
ret=ret.drop(ret.index[:1])

result = pd.concat([df, ret], axis=1, sort=False)
result=result.dropna()

#Renaming columns to make graph more understandable
result.rename(columns={'Adj Close':'Returns'}, inplace=True)
result.rename(columns={'sentimentnum':'Sentiment'}, inplace=True)

#plot returns and sentiment
result.plot(y=['Sentiment', 'Returns'], figsize=(10,5), grid=True)


#3/ Compute the correlation between investor bullishness and stock returns
result['Sentiment'].corr(result['Returns'].shift(1))


#4/ Explore the Granger Causality between investor bullishness and stock returns
from statsmodels.tsa.stattools  import   grangercausalitytests
print(grangercausalitytests(result[['Sentiment', 'Returns']], maxlag=1, addconst=True, verbose=True))

#the null hypothesis would be: X does not granger cause Y or the other way. 
#Also, you accept or reject your null hypothesis depending on the level of significance.
#if P value < Significance level, then Null hypothesis would be rejected.
#if P value > Significance level, then Null hypothesis cannot be rejected.



########OBJECTIVE3
#1/ Implement a machine learning algorithm (Naïve Bayes) 
#to convert each message into a quantitative variable
 
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
cursor = db.stocktwits.find({})
for element in cursor:
    blob = TextBlob(str(element["body"]), analyzer=NaiveBayesAnalyzer())
    sblob  = 0
    if blob.sentiment.classification == "neg":
        sblob = -1
        
    if blob.sentiment.classification == "pos":
        sblob = 1
        
    db.stocktwits.update_one({"id": element["id"]},{'$set': {"sblob": sblob} }, upsert=False)
 
cursor = db.stocktwits.find({},{"created_at":1, "sentimentnum":1,"body":1,"sblob":1})
df2 = pd.DataFrame(list(cursor))
       
#dailyfrequency
df2 = df2.resample("D").mean()
print(df)  

result2 = pd.concat([df2, ret], axis=1, sort=False)
result2= result2.dropna()   

#Compute the correlation between investor bullishness and stock returns
result2['sblob'].corr(result2['Adj Close'].shift(1))

#Explore the Granger Causality between investor bullishness and stock returns
from statsmodels.tsa.stattools  import   grangercausalitytests
print(grangercausalitytests(result2[['sblob', 'Adj Close']], maxlag=1, addconst=True, verbose=True))

    
#2/ Compare the accuracy of a lexicon-based approach with the accuracy of a machine learning method
#create standarized column
cursor = db.stocktwits.find({})
for element in cursor:
    sentimentnum2=0
    if element["sentimentnum"]>0:
        sentimentnum2=1
    if element["sentimentnum"]<0:
        sentimentnum2=-1
    db.stocktwits.update_one({"id": element["id"]},{'$set': {"sentimentnum2": sentimentnum2} }, upsert=False)   

#accuracy for lexicon
cursor = db.stocktwits.find({})
for element in cursor:
    lexiconaccuracy=0
    if element["sentimentnum2"]==element["sentimentnumber"]:
        lexiconaccuracy=1
    db.stocktwits.update_one({"id": element["id"]},{'$set': {"lexiconaccuracy": lexiconaccuracy} }, upsert=False)   

cursor = db.stocktwits.find({},{"created_at":1, "sentimentnum":1,"body":1,"lexiconaccuracy":1})
df3 = pd.DataFrame(list(cursor))
Totallexicon = df3['lexiconaccuracy'].sum()
percentagelexicon= Totallexicon/NUMBErofmessages

#accuracy for Machine Learning
cursor = db.stocktwits.find({})
for element in cursor:
    MLaccuracy=0
    if element["sblob"]==element["sentimentnumber"]:
        MLaccuracy=1
    db.stocktwits.update_one({"id": element["id"]},{'$set': {"MLaccuracy": MLaccuracy} }, upsert=False)   

cursor = db.stocktwits.find({},{"created_at":1, "sentimentnum":1,"body":1,"MLaccuracy":1})
df4 = pd.DataFrame(list(cursor))
TotalML = df4['MLaccuracy'].sum()
percentageML=TotalML/numberofmessages
