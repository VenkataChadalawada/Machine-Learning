#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 17:39:50 2018

@author: vchadalawada
"""

# we are going to use apyori.py in the same directory which we got from python community

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None) #header none as by default pandas makes first line as header but here we dont have any header in data file
#apriori expection a list of lists eg - a list of transaction lists
#transform the dataset

transactions = []
# we need to travers all the rows
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

#training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift = 3, min_length=2) #min_length helps to tell algorithm that it should take minimum 2 transactions in basket to be considered 
# how to compute support , confidence, lift?
# let say you wanna fix with an idea that if an apple purchased 3 times a day which is 7*3 = 21 times a week
# now support would be on our weekly data set = 21/7500 = 0.0028 =~0.003
# similarly 0.2 is a good confidence percentage

#Visualising the results
results = list(rules)
results_list = []
for i in range(0, len(results)):        
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]))

# these rules are sorted