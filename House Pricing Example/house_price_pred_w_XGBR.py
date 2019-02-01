# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:59:59 2019

@author: mazzi
"""

''' 

This code uses the Scikit-Learn Wrapper interface for XGBoost to predict
house prices from the Kaggle competition. It is a supervised learning task
to use features and make a regression to find prices of unlabeled houses.

'''

import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest
from math import sqrt
import numpy as np

# Load data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')


# Encode categorical features
def get_dummies_and_numericals(data):
    
    df_categorical = pd.DataFrame()
    df_numerical = pd.DataFrame()
    
    for col in data.columns:
        # The object type columns have categorical datatype, so we will
        # one-hot encode those columns
        if data[col].dtypes=='object':
            df_categorical = pd.concat((df_categorical,pd.get_dummies(data[col],prefix=col)),
                                       sort=False, axis=1)
        else:
            df_numerical = pd.concat((df_numerical,data[col]),sort=False,axis=1)
    return df_categorical,df_numerical

train_df_categorical, train_df_numerical = get_dummies_and_numericals(train_data)

train = pd.concat((train_df_numerical,train_df_categorical),sort=False,axis=1)
train_col = train.columns
train = train.fillna(0)

# For outlier detection I adatped the code from https://www.kaggle.com/zoupet/neural-network-model-for-house-prices-tensorflow/comments?scriptVersionId=3552772
# That used isolation forest (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
clf = IsolationForest(max_samples =120, random_state = 42, contamination=0.1)
clf.fit(train)
# Create a column with -1 being classified as outlier / 1 not outlier
outlier_col = clf.predict(train)
outlier_col = pd.DataFrame(outlier_col, columns = ['Outlier'])
# Get the index number of rows that are not outliers
outlier_col[outlier_col['Outlier'] == 1].index.values
# Select in train those rows that were classified as non outliers
train = train.iloc[outlier_col[outlier_col['Outlier'] == 1].index.values]
# Reset index
train.reset_index(drop = True, inplace = True)

# Separate the train (with features) and target datasets
target = train['SalePrice']
train = train.drop(['SalePrice'],axis=1)
scaler = StandardScaler()
train = scaler.fit_transform(train)

# Use the same colunms of train to create the test dataframe
X_df = pd.DataFrame(columns=train_col)
# Get categorial and numerical features of test
test_df_categorical, test_df_numerical = get_dummies_and_numericals(test_data)
# Merge categorical and numerical
features_x = pd.concat((test_df_categorical,test_df_numerical),sort=False,axis=1)
# Append to the test dataframe (which contains all columns of train) and remove SalePrice column (which is empty)
X_tst = X_df.append(features_x,sort=False)
X_tst = X_tst.drop('SalePrice',axis=1)
# Fill NaN values
X_tst = X_tst.fillna(0)
# Use the same scaler used in train
X_tst = scaler.transform(X_tst)


# REGRESSOR

X_train, X_valid, y_train, y_valid = train_test_split(list(train), list(target), test_size=0.1, random_state=42)

# Parameters of XGBRegressor
xgb_param = {'tree_method':'gpu_exact','seed':42, 
'max_depth':5,'n_estimators':15000,'learning_rate':0.001,
'min_child_weight':80
}


# fit model no training data
model = XGBRegressor(**xgb_param)
model.fit(X_train, y_train, verbose=True, eval_metric = 'rmse')

# Calculate RMSE of train and validation sets
y_pred_train = model.predict(X_train)
print(sqrt(mean_squared_error(y_train,y_pred_train)))

y_pred_test = model.predict(X_valid)
print(sqrt(mean_squared_error(y_valid,y_pred_test)))


predictions_r = np.exp(model.predict(X_tst))
np.savetxt('preds.csv',predictions_r,delimiter=',')
