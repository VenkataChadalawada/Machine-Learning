# Machine-Learning
## Regression
p-value should be <0.05

### 9 Random Forests
- Its one of the variety of Ensemble learning (it also has gradient boosting etc...)
- pick random k data points in the set then build a decidion tree then choose a number N and build such many trees
then pick a new data point let all your trees answer y point . Now take the average of all.
``` python
#import libraries
#import dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0) #n_estimators=no of trees
regressor.fit(X, y)
# predict
y_pred = regressor.predict(6.5)
```
### 10 Evaluating Regression Models Performance
R Squared :- it tells how good is your best fit line is. R^2 ideal case is 1..ie..if ss-res is zero,
all data points are passing through regression line. we need greater value(upto 1) to get better model. normally its between 0 to 1
- Rsquared = 1- (Sum of squared residuals/Sum of squared totals)
- Residuals is the difference between regression point to actual point
- totals is the difference between average point and actual point
what about multpile linear regression - its similar least sum of squares of residual

Adjusted R2 :- when we add new columns we wont be clearly knowing with R^2 as it will be very tiny when it divides over. hence we need
- adj Rsquared = 1 - (1-R^2) (n-1/n-p-1)
- p is number of regressors , n - sample size.
- This adj R^2 penalizes if we add more independent variables
- Its the indicator when we do backward elimination. The moment it falls and we see not a real big deal in pvalue ~=0.05 and lesser we can take choice

Interpreting coefficients

Regression Model    |    Pros    |    Cons
---------------------------------------------------
Linear Regression   | Works on any size of dataset, gives informations about relevance of features | The Linear Regression Assumptions

Polynomial Regression | Works on any size of dataset, works verywell on non linear problems | Need to choose the right polynomial degree for a good bias/variance tradeoff

SVR     |  Easily adaptable, works very well on non linear problems, not biased by outliers | Compulsory to apply feature scaling, not
well known, more difficult to understand

Decision Tree Regression |  Interpretability, no need for feature scaling, works on both linear / nonlinear problems | Poor results on too small datasets, overfitting can easily occur

Random Forest Regression | Powerful and accurate, good performanceon many problems, including non linear | No interpretability, overfitting can easily occur, need to choose the number of trees

##### How do I know which model to choose for my problem ?

First, you need to figure out whether your problem is linear or non linear. You will learn how to do that in Part 10 - Model Selection. Then:

If your problem is linear, you should go for Simple Linear Regression if you only have one feature, and Multiple Linear Regression if you have several features.

If your problem is non linear, you should go for Polynomial Regression, SVR, Decision Tree or Random Forest. Then which one should you choose among these four ? That you will learn in Part 10 - Model Selection. The method consists of using a very relevant technique that evaluates your models performance, called k-Fold Cross Validation, and then picking the model that shows the best results. Feel free to jump directly to Part 10 if you already want to learn how to do that.

##### How can I improve each of these models ?
In Part 10 - Model Selection, you will find the second section dedicated to Parameter Tuning, that will allow you to improve the performance of your models, by tuning them. You probably already noticed that each model is composed of two types of parameters:

the parameters that are learnt, for example the coefficients in Linear Regression,
the hyperparameters.
The hyperparameters are the parameters that are not learnt and that are fixed values inside the model equations. For example, the regularization parameter lambda or the penalty parameter C are hyperparameters. So far we used the default value of these hyperparameters, and we haven't searched for their optimal value so that your model reaches even higher performance. Finding their optimal value is exactly what Parameter Tuning is about.

### 11 Classification
#### Logistic Regression
ln(p/1-p) =  
Exercise:
``` python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('social_network_ads.csv')
X= dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Logistic Regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# predict the test set results
y_pred = classifier.predict(X_test)

# evaluating using confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
## leftupper+rightbottom tells those many percent is explained by model
## similarly opposite tells how many aren't

# Visualizing the training set results

```
### 13 K-Nearest Neighbors KNN classification
1) choose the number K of neighbors
2) Take the K nearest neighbors of the new data point using Euclidian distance (few others are Manhattan distance etc)
3) Among these K neighbors count the number of data points in each category
4) assign new data point into the most of the neighbors belongs to
Euclidean Distance: - Sqrt((x2-x1)^2 + (y2-y1)^2)

### 14 Support Vector Machine (SVM) classification
- Line is build by a principle called Maximum Margin = sum of Xn and Yn has to be maximized by this line seperation where Xn and Yn are called support vectors.
- line in the middle is called = Maximum Margin Hyper Plane(Maximum margin classifier)
- why SVM is imp?
  - they look for support vectors which are for eg - cats that looks like dogs and dogs that looks like cats in cat dog classification . THis model will be built on these support vectors

### 15 Kernel SVM for non linear data
- with the help of a mapping function, data will be transffered into higher dimensions
- thus we can draw hyperplane in that higher dimensions then classify and then bring the data back to original dimensions
- problem = mapping to higher dimension can be highly compute-intensive, so we will do a kernel-trick
Kernel-Trick: use anyof the kernels if the point is on kernel then its one class and others are in another class
- Gaussian or Radial Basis Functional(RBF) kernel:
- Sigmoid kernel
- polynomial kernel

### 16 Bayes Theorem
P(A|B) = (P(B|A)* P(A)) / P(B)
Machine1: 30 wrenches/hr
Machine2: 20 wrenches/hr
out of all produced parts:
we can see that 1% are defective
out of all defective parts:
we can see that 50% came from machine1 and 50% came from machine 2

Q) what is the probability that a part produced by Machine2 is defective?
P(M1) = 30/50
P(M2) = 20/50
P(Defect)=1%
P(M1|defect) =50%
P(M2|defect) = 50%

P(defect|M2) = P(M2|defect)P(Defect) / P(M2) = (0.5 * 0.01) / 0.4 = 1.25%
also
P(defect|M1) = P(M1|defect)P(Defect) / P(M1) = 0.5 * 0.01 / 0.6 = 
how Naive Bayes applies in classification?
Lets say you have a plot with people walks and people drives to work given their age & salary
lets say anew data point came in now we need Naive bayes to answer which class it could go int
P(Walks|X) = P(X|Walks) * P(walks) / P(X)
prob of walks given a new person = prob person given walks * prob of total walks / probability of x
P(Drives | x) = P(X|Drives) * P(Drives) / P(X)

now compare P(walks|x) vs P(drives|x)

P(walks) = Num of walks / total observation = 10/30 = 0.33
P(x)  = Draw a circle arround the new point with some radius anyone falls =
no of similar observations/ total number of observations = 4/30
P(x|walks) = no of similar observations among those who walk/total number of walkers
= 3/10
P(walks|x) = (3/10 * 10/30) / (4/30) = 0.75
similarly we will find P(drives | x) = 0.25 
p(drives) = no of drives/total num ofobservations = 20/30
p(x) = 4/30 same as aboveP(x)
P(x|drives) = 1/20
P(drives|x) = P(x|drives)
now 0.75 vs 0.25 => 0.75 so It will be "walks" to work


=> Why Naive?
Independence , it assumes age and salary are independent
=> P(x)
likelihood = no of similar observations / total number of observations = 4/30
=> what happens when we have more than 2 classes
compare which one has greater probability
### 17 Decision Trees
It makes splits on the data that creates branches , the split is based on entropy (or Gini or maximum entropy)
They combine with Random Forests, Gradient Boosting etc..
core problem is - it does overfitting most of the time in a complex mixed data set

### 18 Random Forests
Ensemble Learning - using more machine learning algorithms to create a new algorithm
- 1) pick at random K data points from the training set
- 2) Build the Decision tree associated to these K data points
- 3) choose number 'Ntree' of trees you want to build and repeat step1 & 2
- 4) For a new data point, make each one of your Ntree trees predict the category to which the data point belongs and assign the new data point to category that wins.
Imagine 500 decision trees

### 19 Evaluating classification models
#### flasepositives & false negatives
FalsePositive - Type 1 error - we predicted a positive but it was false
False Negative - Type 2 error - we predicted a negative but it effected

Flase positive or type 1 error is kind of warning - you said earthquake gonna come but it didnt
False Negative or type 2 error is kind of dangerous - you said it wont occur but it did

#### confusion matrix
matrix between actual vs predicted
accuracy rate  = correct/total
error rate = wrong/total

#### accuracy paradox 

#### cummilative accuracy profile CAP
how much gain additional gain we get
we can plot diff machine learning models to compare
AR = ar/ap , closer to 1 means better the model
look at the 50th percentile x% percentage
x<60% rubbish
60%<x<70% poor
70% <x<80%  good
80 <x< 90 very good
90 <x< 100 too good ?? over fitting


Classification Model     |     Pros    |     Cons
Logistic Regression Probabilistic approach, gives informations
about statistical significance of features The Logistic Regression Assumptions
K-NN Simple to understand, fast and efficient Need to choose the number of neighbours k
SVM Performant, not biased by outliers,
not sensitive to overfitting
Not appropriate for non linear problems, not
the best choice for large number of features
Kernel SVM High performance on nonlinear problems, not
biased by outliers, not sensitive to overfitting
Not the best choice for large number of
features, more complex
Naive Bayes Efficient, not biased by outliers, works on
nonlinear problems, probabilistic approach
Based on the assumption that features
have same statistical relevance
Decision Tree Classification Interpretability, no need for feature scaling,
works on both linear / nonlinear problems
Poor results on too small datasets,
overfitting can easily occur
Random Forest Classification Powerful and accurate, good performance on
many problems, including non linear
No interpretability, overfitting can easily
occur, need to choose the number of trees


1. What are the pros and cons of each model ?

Please find here a cheat-sheet that gives you all the pros and the cons of each classification model.

2. How do I know which model to choose for my problem ?

Same as for regression models, you first need to figure out whether your problem is linear or non linear. You will learn how to do that in Part 10 - Model Selection. Then:

If your problem is linear, you should go for Logistic Regression or SVM.

If your problem is non linear, you should go for K-NN, Naive Bayes, Decision Tree or Random Forest.

Then which one should you choose in each case ? You will learn that in Part 10 - Model Selection with k-Fold Cross Validation.

Then from a business point of view, you would rather use:

- Logistic Regression or Naive Bayes when you want to rank your predictions by their probability. For example if you want to rank your customers from the highest probability that they buy a certain product, to the lowest probability. Eventually that allows you to target your marketing campaigns. And of course for this type of business problem, you should use Logistic Regression if your problem is linear, and Naive Bayes if your problem is non linear.

- SVM when you want to predict to which segment your customers belong to. Segments can be any kind of segments, for example some market segments you identified earlier with clustering.

- Decision Tree when you want to have clear interpretation of your model results,

- Random Forest when you are just looking for high performance with less need for interpretation. 

3. How can I improve each of these models ?

Same answer as in Part 2: 

In Part 10 - Model Selection, you will find the second section dedicated to Parameter Tuning, that will allow you to improve the performance of your models, by tuning them. You probably already noticed that each model is composed of two types of parameters:

the parameters that are learnt, for example the coefficients in Linear Regression,
the hyperparameters.
The hyperparameters are the parameters that are not learnt and that are fixed values inside the model equations. For example, the regularization parameter lambda or the penalty parameter C are hyperparameters. So far we used the default value of these hyperparameters, and we haven't searched for their optimal value so that your model reaches even higher performance. Finding their optimal value is exactly what Parameter Tuning is about. So for those of you already interested in improving your model performance and doing some parameter tuning, feel free to jump directly to Part 10 - Model Selection

## CLUSTERING
Clustering is similar to classification, but the basis is different. In Clustering you don’t know what you are looking for, and you are trying to identify some segments or clusters in your data. When you use clustering algorithms on your dataset, unexpected things can suddenly pop up like structures, clusters and groupings you would have never thought of otherwise.

In this part, you will understand and learn how to implement the following Machine Learning Clustering models:

- K-Means Clustering
- Hierarchical Clustering

### 21) K-Means clustering
step1 - choose the number k of clusters
step2 - select random k points, the centroids
step3 - assign each data point to the closest centroid -> forms K clusters (based on euclidean distances)
step4 - compute and place new centroid of each cluster
step5 - reassign each data point to the new closest centroid
        if any reassignmnet took place go to step 4 otherwise go to finish
        

prob1 - we need to ensure it picks initial values properly => kmeans++
prob2 - we need to find k value => elbow method analysis

### 22) Hierarchical clustering
Two types
1 - agglomerative (botoom up approach)
2 - divisive (up to bottom)
#### Agglomerative HC:
step1 - make each data point a single point cluster -> that forms N clusters
step2 - Take the two closest data points and make them one cluster -> that forms N-1 clusters
step3 - Take the two closest clusters and make them one cluster -> that forms N-2
step4 - Repeat STEP3 until there is only one cluster
euclidean distance sqrt[ (x2-x1)^2 + (y2-y1)^2 ]

Distance between two clusters : we can find euclidean between below 4 ways
opt1) closest points
opt2)furthest points
opt3)avg distance
opt4)dist between centroids

While forming one giant cluster by Aglomerative HC it stores in memory and creates a Dendogram

#### How do dendograms work?
euclidian distance on vertical axis & points on xaxis
dissimilarity threshold - helps to cut down the vertical axis
recommended take largest distance and consider threshold there


### 23) ASSOCIATIVE RULE LEARNING
People who bought also bought ... That is what Association Rule Learning will help us figure out!

In this part, you will understand and learn how to implement the following Association Rule Learning models:
- Apriori
- Eclat

#### Apriori
Apriori has three parts to it
- support
- confidance
- lift
##### support
examples 
- movie recommendation support(M) = #user watchlists containing Movie / #user watchlists
lets say support for a movie ExMachina = num of perople watched exmachina / total number of people; say for ex = 10/100 = 10%
- market basket optimization(I) = #transactions containing Item / #transactions

#### confidance
people who have seen interstellar also would see ExMachina
confidance(M1->M2) = #user watchlists containing M1 and M2 / #user watchlists containing M1

#### Lift
confidance(M1->M2) / Support(M)

#### Algorithm
- step1 set a minimum support and confidence
- step2 take all the subsets in transactions having higher support than minimum support
- step3 take all the rules of these subsets having higher confidance than minimum confidence
- step4 sort the rules by decreasing lift

#### Eclat Model
in this model we only have 'Support'
- eclat support => support(M) = #user watchlists containing M /  # user watchlists; here M is a set of 2 or more movies
##### Algorithm
- step 1: set a minimum support
- step 2: Take all the transactions having higher support than minimum support
- step 3: sort these subsets by decreasing support

### PART 6 - REINFORCEMENT LEARNING
Reinforcement Learning is a branch of Machine Learning, also called Online Learning. It is used to solve interacting problems where the data observed up to time t is considered to decide which action to take at time t + 1. It is also used for Artificial Intelligence when training machines to perform tasks such as walking. Desired outcomes provide the AI with reward, undesired with punishment. Machines learn through trial and error.

In this part, you will understand and learn how to implement the following Reinforcement Learning models:

Upper Confidence Bound (UCB)
Thompson Sampling

#### 27 Upper confidence Bound

##### Multi Armed Bandit Problem

##### 28 Thompson Sampling

### Part 7 - Natural Language Processing
Natural Language Processing (or NLP) is applying Machine Learning models to text and language. Teaching machines to understand what is said in spoken and written word is the focus of Natural Language Processing. Whenever you dictate something into your iPhone / Android device that is then converted to text, that’s an NLP algorithm in action.

You can also use NLP on a text review to predict if the review is a good one or a bad one. You can use NLP on an article to predict some categories of the articles you are trying to segment. You can use NLP on a book to predict the genre of the book. And it can go further, you can use NLP to build a machine translator or a speech recognition system, and in that last example you use classification algorithms to classify language. Speaking of classification algorithms, most of NLP algorithms are classification models, and they include Logistic Regression, Naive Bayes, CART which is a model based on decision trees, Maximum Entropy again related to Decision Trees, Hidden Markov Models which are models based on Markov processes.

A very well-known model in NLP is the Bag of Words model. It is a model used to preprocess the texts to classify before fitting the classification algorithms on the observations containing the texts.

In this part, you will understand and learn how to:

Clean texts to prepare them for the Machine Learning models,
Create a Bag of Words model,
Apply Machine Learning models onto this Bag of Worlds model.
