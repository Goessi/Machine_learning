## 2. Introduction To The Data ##

import pandas as pd
data = pd.read_csv('AmesHousing.txt',delimiter = '\t')
train = data.iloc[:1460]
test = data.iloc[1460:]
print(data.info())
target = 'SalePrice'

## 3. Simple Linear Regression ##

import matplotlib.pyplot as plt
# For prettier plots.
import seaborn

fig = plt.figure()
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)
ax1.scatter(train['Garage Area'],train['SalePrice'])
ax2.scatter(train['Gr Liv Area'],train['SalePrice'])
ax3.scatter(train['Overall Cond'],train['SalePrice'])
plt.show()

## 5. Using Scikit-Learn To Train And Predict ##

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train[['Gr Liv Area']],train['SalePrice'])
print(lr.coef_)
print(lr.intercept_)
a1 = lr.coef_
a0 = lr.intercept_

## 6. Making Predictions ##

import numpy as np
from sklearn.metrics import mean_squared_error
lr = LinearRegression()
lr.fit(train[['Gr Liv Area']], train['SalePrice'])
pre1 = lr.predict(train[['Gr Liv Area']])
pre2 = lr.predict(test[['Gr Liv Area']])
train_rmse = (mean_squared_error(pre1,train['SalePrice']))**0.5
test_rmse = (mean_squared_error(pre2,test['SalePrice']))**0.5

## 7. Multiple Linear Regression ##

cols = ['Overall Cond', 'Gr Liv Area']
lr = LinearRegression()
lr.fit(train[cols],train['SalePrice'])
pre1 = lr.predict(train[cols])
pre2 = lr.predict(test[cols])
train_rmse_2 = (mean_squared_error(pre1,train['SalePrice']))**0.5
test_rmse_2 = (mean_squared_error(pre2,test['SalePrice']))**0.5
