#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 19:56:37 2018

@author: vchadalawada
"""

# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocesiing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding Catgeorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1]) #since first row countries has more than 2 categories
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] # we will remove to get rid of dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling ? Do we need for ANN yes 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ------- Part 2 - Now lets make ANN ----------
# Importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# initialize weights will take care by dense , we will use rectifier act func for hidden layers & sigmoid for o/p layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11 ))
#output_dim is how many hidden layer nodes => 2 ways a) trial & error b) average input +outgput nodes (11 + 1) / 2 = 6
# init takes weights close to 0 and uniformly distributed
# activation => activation function; relu for Rectifier
#input_dim => input nodes

#second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

#output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid' ))

#compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics =['accuracy'] )
# if output is more than 2 loss would be category_crossentropy


# Fitting the ANN to the Training Set
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# accuracy = no of correct / tot no of predictions 1873+264 / 1873+264+118+245 = 2137/2500 = 0.8548