## 1. Introduction ##

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('AmesHousing.txt', delimiter="\t")
train = data[0:1460]
test = data[1460:]

features = ['Wood Deck SF', 'Fireplaces', 'Full Bath', '1st Flr SF', 'Garage Area',
       'Gr Liv Area', 'Overall Qual']
X = train[features]
y = train['SalePrice']
# remember use np.dot() to calculate

cal = np.dot(np.transpose(X),X)
cal = np.linalg.inv(cal)
cal = np.dot(cal,np.transpose(X))
a = np.dot(cal,y)
