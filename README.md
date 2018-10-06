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


