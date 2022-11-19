# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 18:37:53 2022

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel('G:/Project/dataset_project.xlsx')
print(data)
data = data.drop('EducationField', axis=1)
data = data.drop('Unnamed: 0', axis=1)
data = data.drop('JobRole', axis=1) 
data.columns
x= data.iloc [:, : -1] # ” : ” means it will select all rows,    “: -1 ” means that it will ignore last column
y= data.iloc [:, -1 :] # ” : ” means it will select all rows,    “-1 : ” means that it will ignore all columns 


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=20, max_features="auto", random_state=0)
rf_model.fit(x, y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

y_pred = rf_model.predict(x_test)
x_pred = rf_model.predict(x_train)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import mean_squared_error, r2_score

# r2 score on test dataset
r2_score(y_test, y_pred)

# r2 score on train dataset
r2_score(y_train, x_pred)  

import pickle
import joblib
with open('rf_model.pkl', 'wb') as model_file: 
    pickle.dump(rf_model, model_file)
model = joblib.load('rf_model.pkl')








