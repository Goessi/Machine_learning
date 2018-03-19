## 2. Introduction to the data ##

import pandas as pd
dc_listings = pd.read_csv('dc_airbnb.csv')
print(dc_listings.iloc[0])

## 4. Euclidean distance ##

import numpy as np
real = dc_listings.iloc[0]['accommodates']
first_distance = abs(real - 3)
print(first_distance)

## 5. Calculate distance for all observations ##

dc_listings['distance'] = dc_listings['accommodates'].apply(lambda x:np.abs(x - 3))
print(dc_listings['distance'].value_counts())

## 6. Randomizing, and sorting ##

import numpy as np
np.random.seed(1)
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]
dc_listings = dc_listings.sort_values('distance')
print(dc_listings['price'][:10])

## 7. Average price ##

stripped_commas = dc_listings['price'].str.replace(',','')
stripped_money = stripped_commas.str.replace('$','')
dc_listings['price'] = stripped_money.astype(float)
mean_price = np.mean(dc_listings[:5])
print(mean_price)

## 8. Function to make predictions ##

# Brought along the changes we made to the `dc_listings` Dataframe.
dc_listings = pd.read_csv('dc_airbnb.csv')
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]

def predict_price(new_listing):
    temp_df = dc_listings.copy()
    temp_df['distance'] = temp_df['accommodates'].apply(lambda x: np.abs(x - new_listing))
    price = np.mean(temp_df.sort_values('distance')['price'][:5])
    return price

acc_one = predict_price(1)
acc_two = predict_price(2)
acc_four = predict_price(4)
