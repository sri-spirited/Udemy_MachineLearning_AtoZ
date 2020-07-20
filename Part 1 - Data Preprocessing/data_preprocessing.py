# -*- coding: utf-8 -*-
"""
Created on Thu May 24 22:38:27 2018

@author: sridevi.tolety
"""

#Importing the libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import Imputer

#Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values # This is an object, not a dataframe
X
y = dataset.iloc[:,3].values
y

# Missing data 
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
dir(imputer)
imputer = imputer.fit(X[:,1:3]) # Fit only on the second and third columns (index 1 and 2) 
X[:,1:3] = imputer.transform(X[:,1:3]) # Replace original columns with imputed columns
X

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
LabelEncoder_X = LabelEncoder()
LabelEncoder_X.fit_transform(X[:,0]) #Fit to country and returns country encoded 
#array([0, 2, 1, 2, 1, 0, 2, 0, 1, 0], dtype=int64) 
# So fit these encoded values in place of country
X[:,0] = LabelEncoder_X.fit_transform(X[:,0])
X
#Problem : the labels 0, 1, 2 might be interpreted by the machine to be ordered. 
# One hot encoding instead 
ohe = OneHotEncoder(categorical_features=[0]) #categorical_features=[0] means first column to be OHE
X = ohe.fit_transform(X).toarray()
X
# For dependent variable, can do label encoding because n=only 2 categories 
LabelEncoder_y = LabelEncoder()
y = LabelEncoder_y.fit_transform(y)
y


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) # Dont fit to test set, already fit on train 

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

