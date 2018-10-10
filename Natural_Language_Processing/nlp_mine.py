#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 20:54:31 2018

@author: vchadalawada
"""
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing 
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3) #quoting =3 ignore double quotes

#cleaning the texts 
import re
import nltk
nltk.download('stopwords') #just to download
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000): #since we have 999 reviews
    review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][i])  #only keep letters => remove punctuation numbers
    review = review.lower()  #convert everything into lower case letters
     #remove non significant words => from stop words
    review = review.split() #first converting into words
    review = [word for word in review if not word in set(stopwords.words('english'))] #filtering out stop words
    #Stemming = keeping same word for different forms of same root for eg- loved => love
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review]
    # we can combine [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# creating the Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #1500 top featured out of 1565 we got with out max_features
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


#Fitting classifier to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
