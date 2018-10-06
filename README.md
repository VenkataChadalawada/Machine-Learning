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
