## 1. Introduction ##

import numpy as np
import pandas as pd

dc_listings = pd.read_csv("dc_airbnb.csv")
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')
shuffled_index = np.random.permutation(dc_listings.index)
dc_listings = dc_listings.reindex(shuffled_index)
split_one = dc_listings.iloc[:1862]
split_two = dc_listings.iloc[1862:]

## 2. Holdout Validation ##

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

train_one = split_one
test_one = split_two
train_two = split_two
test_two = split_one

knn = KNeighborsRegressor()
knn.fit(train_one[['accommodates']],train_one['price'])
pre = knn.predict(test_one[['accommodates']])
mse = mean_squared_error(pre,test_one['price'])
iteration_one_rmse = mse**(1/2)

knn = KNeighborsRegressor()
knn.fit(train_two[['accommodates']],train_two['price'])
pre = knn.predict(test_two[['accommodates']])
mse = mean_squared_error(pre,test_two['price'])
iteration_two_rmse = mse**(1/2)
avg_rmse = np.mean([iteration_one_rmse,iteration_two_rmse])

## 3. K-Fold Cross Validation ##

dc_listings['fold'] = float(0)
dc_listings['fold'][0:744] = 1
dc_listings['fold'][744:1488] = 2
dc_listings['fold'][1488:2232] = 3
dc_listings['fold'][2232:2976] = 4
dc_listings['fold'][2976:3723] = 5
print(dc_listings['fold'].value_counts())


## 4. First iteration ##

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
train = dc_listings[dc_listings['fold'] != 1]
test = dc_listings[dc_listings['fold'] == 1]
knn = KNeighborsRegressor()
knn.fit(train[['accommodates']],train['price'])
labels = knn.predict(test[['accommodates']])
mse = mean_squared_error(labels,test['price'])
iteration_one_rmse = mse**(1/2)

## 5. Function for training models ##

# Use np.mean to calculate the mean.
import numpy as np
fold_ids = [1,2,3,4,5]


def train_and_validate(df,folds):
    rmse_l = list()
    for v in folds:
        knn = KNeighborsRegressor()
        train = df[df['fold'] != v]
        test = df[df['fold'] == v]
        knn = KNeighborsRegressor()
        knn.fit(train[['accommodates']],train['price'])
        pre = knn.predict(test[['accommodates']])
        mse = mean_squared_error(pre,test['price'])
        rmse = mse**(1/2)
        rmse_l.append(rmse)
    return rmse_l
rmses = train_and_validate(dc_listings,fold_ids)
avg_rmse = sum(rmses)/len(rmses)
print(rmses)
print(avg_rmse)

## 6. Performing K-Fold Cross Validation Using Scikit-Learn ##

from sklearn.model_selection import cross_val_score, KFold
kf = KFold(n_splits = 5,shuffle = True,random_state = 1)
knn = KNeighborsRegressor()
mses = cross_val_score(knn,dc_listings[['accommodates']],dc_listings['price'],scoring='neg_mean_squared_error',cv=kf)
s = 0
for v in mses:
    v = (abs(v))**(1/2)
    s = s + v
avg_rmse = s/len(mses)

## 7. Exploring Different K Values ##

from sklearn.model_selection import cross_val_score, KFold

num_folds = [3, 5, 7, 9, 10, 11, 13, 15, 17, 19, 21, 23]

for fold in num_folds:
    kf = KFold(fold, shuffle=True, random_state=1)
    model = KNeighborsRegressor()
    mses = cross_val_score(model, dc_listings[["accommodates"]], dc_listings["price"], scoring="neg_mean_squared_error", cv=kf)
    rmses = np.sqrt(np.absolute(mses))
    avg_rmse = np.mean(rmses)
    std_rmse = np.std(rmses)
    print(str(fold), "folds: ", "avg RMSE: ", str(avg_rmse), "std RMSE: ", str(std_rmse))
