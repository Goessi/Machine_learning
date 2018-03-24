## 1. Introduction ##

import pandas as pd

data = pd.read_csv('AmesHousing.txt', delimiter="\t")
train = data[0:1460]
test = data[1460:]

train_null_counts = train.isnull().sum()
print(train_null_counts)

df_no_mv = train[train_null_counts[train_null_counts == 0].index]

## 2. Categorical Features ##

text_cols = df_no_mv.select_dtypes(include=['object']).columns

for col in text_cols:
    print(col+":", len(train[col].unique()))
    train[col] = train[col].astype('category')
    
print(train['Utilities'].cat.codes.value_counts())


## 3. Dummy Coding ##

dummy_cols = pd.DataFrame()
for col in text_cols:
    dummy_cols = pd.get_dummies(train[col])
    train = pd.concat([train,dummy_cols],axis = 1)
    train = train.drop(col,1)

## 4. Transforming Improper Numerical Features ##

train['years_until_remod'] = train['Year Remod/Add'] - train['Year Built']

## 5. Missing Values ##

import pandas as pd

data = pd.read_csv('AmesHousing.txt', delimiter="\t")
train = data[0:1460]
test = data[1460:]

train_null_counts = train.isnull().sum()
cols = train_null_counts.index
df_missing_values = pd.DataFrame()
for col in cols:
    if train_null_counts[col]>0 and train_null_counts[col]<584:
        df_missing_values[col] = train[col]
print(df_missing_values.isnull().sum())
print(df_missing_values.dtypes)

## 6. Imputing Missing Values ##

float_cols = df_missing_values.select_dtypes(include=['float'])
float_cols = float_cols.fillna(float_cols.mean())
print(float_cols.isnull().sum())
