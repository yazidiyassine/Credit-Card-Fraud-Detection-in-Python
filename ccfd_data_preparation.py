# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:19:52 2023

@author: Yassine Yazidi
"""
import pandas as pd


dataset = pd.read_csv("C:\\Users\\Yassine Yazidi\\OneDrive\\Desktop\\Testing\\data\\creditcard.csv")

# =============================================================================
# 
#      Data Preparation
# 
# =============================================================================

# Check for missing|null values
dataset.isnull().shape[0]
print("None-missing values: "+str(dataset.isnull().shape[0]))
print("Misssing values: "+ str(dataset.shape[0] - dataset.isnull().shape[0]))

""" As there are no missing data, we turn to standardization.
We standardize only Time and Amount using RobustScaler:
 """
from sklearn.preprocessing import RobustScaler
""" This Scaler removes the median and scales the data
according to the quantile range (defaults to IQR:
Interquartile Range). The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile). """
scaler = RobustScaler().fit(dataset[["Time", "Amount"]])
dataset[["Time", "Amount"]] = scaler.transform(dataset[["Time", "Amount"]])

print(dataset.head().append(dataset.tail())) # type: ignore

"""Dividing the data into features and targets.
Making the train-test split of the data: """

# Separate response and features underSampling before cross validation will lead to overfitting 
y = dataset["Class"] # target
x = dataset.iloc[:, 0:30]

# Use sklearn for splitting the data
from sklearn.model_selection import train_test_split
X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(X_train.shape, x_test.shape, Y_train.shape, y_test.shape)