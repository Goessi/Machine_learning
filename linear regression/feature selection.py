## 1. Missing Values ##

import pandas as pd
data = pd.read_csv('AmesHousing.txt', delimiter="\t")
train = data[0:1460]
test = data[1460:]
numerical_train = train.select_dtypes(include=['int', 'float'])
cols = ['PID','Year Built','Year Remod/Add','Garage Yr Blt','Mo Sold','Yr Sold']
numerical_train = numerical_train.drop(cols,axis=1)
null_series = numerical_train.isnull().sum()
full_cols_series = null_series[null_series == 0]
print(full_cols_series)

## 2. Correlating Feature Columns With Target Column ##

train_subset = train[full_cols_series.index]
print(train_subset.info())
sorted_corrs = abs(train_subset.corr()['SalePrice']).sort_values()

## 3. Correlation Matrix Heatmap ##

import seaborn as sns
import matplotlib.pyplot as plt
strong_corrs = sorted_corrs[sorted_corrs > 0.3]
matrix = train_subset[strong_corrs.index].corr()
sns.heatmap(matrix)

## 4. Train And Test Model ##

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

final_corr_cols = strong_corrs.drop(['Garage Cars', 'TotRms AbvGrd'])
features = final_corr_cols.drop(['SalePrice']).index
target = 'SalePrice'
clean_test = test[final_corr_cols.index].dropna()
lr = LinearRegression()
lr.fit(train[features],train[target])
pre_test = lr.predict(clean_test[features])
pre_train = lr.predict(train[features])
test_rmse = mean_squared_error(pre_test,clean_test['SalePrice'])**0.5
train_rmse = mean_squared_error(pre_train,train['SalePrice'])**0.5

## 5. Removing Low Variance Features ##

train = train[features]
sorted_vars = []
max_v = train.max()
min_v = train.min()
df = (train - min_v)/(max_v - min_v)
sorted_vars = df.var().sort_values()
print(sorted_vars)

## 6. Final Model ##

features = features.drop('Open Porch SF')
lr = LinearRegression()
lr.fit(train[features],train['SalePrice'])
pre_test = lr.predict(clean_test[features])
pre_train = lr.predict(train[features])
train_rmse_2 = mean_squared_error(pre_train,train['SalePrice'])**0.5
test_rmse_2 = mean_squared_error(pre_test,clean_test['SalePrice'])**0.5
print(train_rmse_2)
print(test_rmse_2)
