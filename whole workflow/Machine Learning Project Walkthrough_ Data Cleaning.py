## 3. Reading in to Pandas ##

import pandas as pd
loans_2007 = pd.read_csv('loans_2007.csv')
print(loans_2007.iloc[0])
print(len(loans_2007.columns.values))

## 5. First group of columns ##

cols = ['id','member_id','funded_amnt','funded_amnt_inv','grade','sub_grade','emp_title','issue_d']
loans_2007 = loans_2007.drop(cols, axis=1)

## 7. Second group of features ##

cols = ['zip_code','out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp']
loans_2007 = loans_2007.drop(cols,axis=1)

## 9. Third group of features ##

cols = ['total_rec_int','total_rec_late_fee','recoveries',
'collection_recovery_fee','last_pymnt_d','last_pymnt_amnt']
loans_2007 = loans_2007.drop(cols, axis=1)
print(loans_2007.iloc[0])
print(loans_2007.shape[1])

## 10. Target column ##

print(loans_2007['loan_status'].value_counts())

## 12. Binary classification ##

loans_2007 = loans_2007[(loans_2007['loan_status'] == 'Fully Paid')|(loans_2007['loan_status'] == 'Charged Off')]
replace = {'loan_status':{'Fully Paid':1, 'Charged Off':0}}
loans_2007 = loans_2007.replace(replace)

## 13. Removing single value columns ##

drop_columns = []
cols = loans_2007.columns.values
for col in cols:
    non_null = loans_2007[col].dropna()
    unique_non_null = non_null.unique()
    if len(unique_non_null) == 1:
        drop_columns.append(col)
loans_2007 = loans_2007.drop(drop_columns, axis=1)
print(drop_columns)