## 1. Introduction ##

import pandas as pd
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]
cars = pd.read_table("auto-mpg.data", delim_whitespace=True, names=columns)
filtered_cars = cars[cars['horsepower'] != '?']
filtered_cars['horsepower'] = filtered_cars['horsepower'].astype('float')

## 3. Bias-variance tradeoff ##

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def train_and_test(cols):
    lr = LinearRegression()
    lr.fit(filtered_cars[cols],filtered_cars['mpg'])
    pre = lr.predict(filtered_cars[cols])
    var = np.var(pre)
    mse = mean_squared_error(pre, filtered_cars['mpg'])
    return(mse,var)
cyl_mse,cyl_var = train_and_test(['cylinders'])
weight_mse,weight_var = train_and_test(['weight'])

## 4. Multivariate models ##

# Our implementation for train_and_test, takes in a list of strings.
def train_and_test(cols):
    # Split into features & target.
    features = filtered_cars[cols]
    target = filtered_cars["mpg"]
    # Fit model.
    lr = LinearRegression()
    lr.fit(features, target)
    # Make predictions on training set.
    predictions = lr.predict(features)
    # Compute MSE and Variance.
    mse = mean_squared_error(filtered_cars["mpg"], predictions)
    variance = np.var(predictions)
    return(mse, variance)

one_mse, one_var = train_and_test(["cylinders"])
two_mse, two_var = train_and_test(['cylinders','displacement'])
three_mse, three_var = train_and_test(['cylinders','displacement','horsepower'])
four_mse, four_var = train_and_test(['cylinders','displacement','horsepower','weight'])
five_mse,five_var = train_and_test(['cylinders','displacement','horsepower','weight','acceleration'])
six_mse, six_var = train_and_test(['cylinders','displacement','horsepower','weight','acceleration','model year'])
seven_mse, seven_var = train_and_test(['cylinders','displacement','horsepower','weight','acceleration','model year','origin'])
print(one_mse,two_mse,three_mse,four_mse,five_mse,six_mse,seven_mse)

## 5. Cross validation ##

from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

def train_and_cross_val(cols):
    mse_values = []
    var_values = []
    features = filtered_cars[cols]
    target = filtered_cars['mpg']
    
    kf = KFold(n=filtered_cars.shape[0],n_folds=10,shuffle=True,random_state=3)
    
    for train_index, test_index in kf:
        X_train = features.iloc[train_index]
        X_target = target.iloc[train_index]
        
        y_train = features.iloc[test_index]
        y_target = target.iloc[test_index]
        
        lr = LinearRegression()
        lr.fit(X_train,X_target)
        pre = lr.predict(y_train)
        
        mse = mean_squared_error(pre, y_target)
        var = np.var(pre)
        
        mse_values.append(mse)
        var_values.append(var)
   
    avg_mse = np.mean(mse_values)
    avg_var = np.mean(var_values)
    return (avg_mse,avg_var)

two_mse, two_var = train_and_cross_val(['cylinders','displacement'])
three_mse, three_var = train_and_cross_val(['cylinders','displacement','horsepower'])
four_mse, four_var = train_and_cross_val(['cylinders','displacement','horsepower','weight'])
five_mse, five_var = train_and_cross_val(['cylinders','displacement','horsepower','weight','acceleration'])
six_mse, six_var = train_and_cross_val(['cylinders','displacement','horsepower','weight','acceleration','model year'])
seven_mse, seven_var = train_and_cross_val(['cylinders','displacement','horsepower','weight','acceleration','model year','origin'])
print(two_mse,three_mse,four_mse,five_mse,six_mse,seven_mse)
print(two_var,three_var,four_var,five_var,six_var,seven_var)

## 6. Plotting cross-validation error vs. cross-validation variance ##

import matplotlib.pyplot as plt
        
two_mse, two_var = train_and_cross_val(["cylinders", "displacement"])
three_mse, three_var = train_and_cross_val(["cylinders", "displacement", "horsepower"])
four_mse, four_var = train_and_cross_val(["cylinders", "displacement", "horsepower", "weight"])
five_mse, five_var = train_and_cross_val(["cylinders", "displacement", "horsepower", "weight", "acceleration"])
six_mse, six_var = train_and_cross_val(["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year"])
seven_mse, seven_var = train_and_cross_val(["cylinders", "displacement", "horsepower", "weight", "acceleration","model year", "origin"])

plt.scatter([2,3,4,5,6,7],[two_mse,three_mse,four_mse,five_mse,six_mse,seven_mse],color = 'red')
plt.scatter([2,3,4,5,6,7],[two_var,three_var,four_var,five_var,six_var,seven_var],color = 'blue')
plt.show()