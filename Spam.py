# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 12:19:04 2020

@author: HI
"""
#loading the dataset and import packages
import pandas as pd
dataset = pd.read_csv('datasets_483_982_spam.csv',encoding='latin-1')
dataset = dataset.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis = 1)
dataset = dataset.rename(columns = {'v1': 'Label', 'v2':'Message'})
#data preprocessing and data cleaning
dataset.Label = dataset.Label.map({'spam':1,'ham':0}).astype(int)
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wt = WordNetLemmatizer()
corpus = []
for i in range(0, len(dataset)):
    text  = re.sub('[^a-z,A-Z]', ' ', dataset['Message'][i])
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    text = ''.join(text)
    corpus.append(text)
    
#applying bag of words or tdidf
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
#using tfidf
# from sklearn.feature_extraction.text import TfidfVectorizer
# tf = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()   
y=pd.get_dummies(dataset['Label'])
y=y.iloc[:,1].values
from sklearn.model_selection import  train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0,test_size = 0.33)
# Training model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred=spam_detect_model.predict(X_test)
print(y_pred)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

