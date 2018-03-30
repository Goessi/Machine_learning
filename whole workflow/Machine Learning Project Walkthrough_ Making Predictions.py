## 1. Recap ##

import pandas as pd
loans = pd.read_csv('cleaned_loans_2007.csv')
print(loans.info())

## 3. Picking an error metric ##

import pandas as pd
tn_filter = (predictions == 0)&(loans['loan_status'] == 0)
tp_filter = (predictions == 1)&(loans['loan_status'] == 1)
fn_filter = (predictions == 0)&(loans['loan_status'] == 1)
fp_filter = (predictions == 1)&(loans['loan_status'] == 0)
tn = predictions[tn_filter].shape[0]
tp = predictions[tp_filter].shape[0]
fn = predictions[fn_filter].shape[0]
fp = predictions[fp_filter].shape[0]

## 5. Class imbalance ##

import pandas as pd
import numpy

# Predict that all loans will be paid off on time.
predictions = pd.Series(numpy.ones(loans.shape[0]))
fp_filter = (predictions == 1)&(loans['loan_status'] == 0)
tn_filter = (predictions == 0)&(loans['loan_status'] == 0)
fn_filter = (predictions == 0)&(loans['loan_status'] == 1)
tp_filter = (predictions == 1)&(loans['loan_status'] == 1)
fp = len(predictions[fp_filter])
tn = len(predictions[tn_filter])
fn = len(predictions[fn_filter])
tp = len(predictions[tp_filter])
fpr = fp/(fp+tn)
tpr = tp/(fn + tp)

## 6. Logistic Regression ##

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
target = loans['loan_status']
features = loans.drop(['loan_status'], axis=1)
lr.fit(features, target)
predictions = lr.predict(features)

## 7. Cross Validation ##

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_predict, KFold
lr = LogisticRegression()
kf = KFold(features.shape[0], random_state=1)
predictions = cross_val_predict(lr, features, target, cv=kf)
predictions = pd.Series(predictions)
fn_filter = (predictions == 0)&(target == 1)
tp_filter = (predictions == 1)&(target == 1)
fp_filter = (predictions == 1)&(target == 0)
tn_filter = (predictions == 0)&(target == 0)
fn = len(predictions[fn_filter])
tp = len(predictions[tp_filter])
fp = len(predictions[fp_filter])
tn = len(predictions[tn_filter])
tpr = tp/(tp + fn)
fpr = fp/(fp + tn)
print(tpr)
print(fpr)

## 9. Penalizing the classifier ##

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_predict
lr = LogisticRegression(class_weight='balanced')
kf = KFold(features.shape[0],random_state=1)
predictions = cross_val_predict(lr, features, target,cv=kf)
predictions = pd.Series(predictions)
fn_filter = (predictions == 0)&(target == 1)
tp_filter = (predictions == 1)&(target == 1)
fp_filter = (predictions == 1)&(target == 0)
tn_filter = (predictions == 0)&(target == 0)
fn = predictions[fn_filter].shape[0]
tp = predictions[tp_filter].shape[0]
fp = predictions[fp_filter].shape[0]
tn = predictions[tn_filter].shape[0]
tpr = tp/(tp + fn)
fpr = fp/(fp + tn)
print(tpr)
print(fpr)

## 10. Manual penalties ##

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_predict
penalty = {0:10, 1:1}
lr = LogisticRegression(class_weight=penalty)
kf = KFold(features.shape[0],random_state=1)
predictions = cross_val_predict(lr, features, target,cv=kf)
predictions = pd.Series(predictions)
fn_filter = (predictions == 0)&(target == 1)
tp_filter = (predictions == 1)&(target == 1)
fp_filter = (predictions == 1)&(target == 0)
tn_filter = (predictions == 0)&(target == 0)
fn = predictions[fn_filter].shape[0]
tp = predictions[tp_filter].shape[0]
fp = predictions[fp_filter].shape[0]
tn = predictions[tn_filter].shape[0]
tpr = tp/(tp + fn)
fpr = fp/(fp + tn)
print(tpr)
print(fpr)

## 11. Random forests ##

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_predict
rfc = RandomForestClassifier(random_state=1,class_weight='balanced')
kf = KFold(features.shape[0],random_state=1)
predictions = cross_val_predict(rfc, features, target,cv=kf)
predictions = pd.Series(predictions)
fn_filter = (predictions == 0)&(target == 1)
tp_filter = (predictions == 1)&(target == 1)
fp_filter = (predictions == 1)&(target == 0)
tn_filter = (predictions == 0)&(target == 0)
fn = predictions[fn_filter].shape[0]
tp = predictions[tp_filter].shape[0]
fp = predictions[fp_filter].shape[0]
tn = predictions[tn_filter].shape[0]
tpr = tp/(tp + fn)
fpr = fp/(fp + tn)
print(tpr)
print(fpr)