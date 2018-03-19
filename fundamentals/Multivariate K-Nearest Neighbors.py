## 1. Recap ##

import pandas as pd
import numpy as np
np.random.seed(1)

dc_listings = pd.read_csv('dc_airbnb.csv')
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')
print(dc_listings.info())

## 2. Removing features ##

cols = ['room_type','city','state','latitude','longitude','zipcode','host_response_rate','host_acceptance_rate','host_listings_count']
dc_listings = dc_listings.drop(cols,axis=1)

## 3. Handling missing values ##

cols = ['cleaning_fee','security_deposit']
dc_listings = dc_listings.drop(cols,axis=1)
dc_listings = dc_listings.dropna(axis=0)
print(dc_listings.isnull().sum())

## 4. Normalize columns ##

normalized_listings = (dc_listings - dc_listings.mean())/(dc_listings.std())
normalized_listings['price'] = dc_listings['price']
print(normalized_listings.iloc[:3])

## 5. Euclidean distance for multivariate case ##

from scipy.spatial import distance
row1 = [normalized_listings['accommodates'].iloc[0],normalized_listings['bathrooms'].iloc[0]]
row2 = [normalized_listings['accommodates'].iloc[4],normalized_listings['bathrooms'].iloc[4]]
first_fifth_distance = distance.euclidean(row1,row2)
print(first_fifth_distance)

## 7. Fitting a model and making predictions ##

from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5,algorithm='brute')

train_df = normalized_listings.iloc[0:2792]
test_df = normalized_listings.iloc[2792:]
feature = train_df[['accommodates','bathrooms']]
target = train_df['price']
knn.fit(feature,target) # train the model
predictions = knn.predict(test_df[['accommodates','bathrooms']])# use the model

## 8. Calculating MSE using Scikit-Learn ##

from sklearn.metrics import mean_squared_error

train_columns = ['accommodates', 'bathrooms']
knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute', metric='euclidean')
knn.fit(train_df[train_columns], train_df['price'])
predictions = knn.predict(test_df[train_columns])
two_features_mse = mean_squared_error(test_df['price'],predictions)
two_features_rmse = two_features_mse**(1/2)
print(two_features_mse)
print(two_features_rmse)

## 9. Using more features ##

features = ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute')
knn.fit(train_df[features],train_df['price'])
four_predictions = knn.predict(test_df[features])
four_mse = mean_squared_error(test_df['price'],four_predictions)
four_rmse = four_mse**(1/2)
print(four_mse)
print(four_rmse)


## 10. Using all features ##

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
features = list(train_df.drop(['price'],axis=1))
knn = KNeighborsRegressor(algorithm='brute')
knn.fit(train_df[features],train_df['price'])
all_features_predictions = knn.predict(test_df[features])
all_features_mse = mean_squared_error(test_df['price'],all_features_predictions)
all_features_rmse = all_features_mse**(1/2)
print(all_features_mse)
print(all_features_rmse)
