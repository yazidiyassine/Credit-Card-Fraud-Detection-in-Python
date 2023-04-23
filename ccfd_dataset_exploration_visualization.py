# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:19:52 2023

@author: Yassine Yazidi
"""


# Importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec



# =============================================================================
# # read the dataset
# The dataset contains information on transactions, including the time,
# amount, and whether or not the transaction was fraudulent.
# =============================================================================

# =============================================================================
# # Download the dataset from the URL and save it to a file
"""!! url = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download"""
# =============================================================================

# Read the dataset from the file
dataset = pd.read_csv("creditcard.csv")
"""
    ¤ Time :is measured in seconds since the first 
        transaction in the data collection.
    ¤ Class : the response variable, and it has
                a value of 1 if there is fraud
                    and 0 otherwise.
    
"""

# read the first 5 and last 5 rows of the data
#print(dataset.head().append(dataset.tail()))


# =============================================================================
# 
#     Data Exploration and Visualization
# 
# =============================================================================
# Check for relative proportion

print("Fraudulent Cases: "+str(len(dataset[dataset["Class"] == 1])))
print("Valid Transactions: "+str(len(dataset[dataset["Class"] == 0])))
print("Proprotion Fraudulent Cases: "+str(len(dataset[dataset["Class"] == 1])/ dataset.shape[0]))


# To see how small are the number of Fraud transactions
data_p = dataset.copy()
data_p[" "] = np.where(data_p["Class"] == 1, "Fraud", "Genuine")


# Plot a pie chart
""" There is an imbalance in the data, with only 0.17% of the total cases being fraudulent."""
data_p[" "].value_counts().plot(kind="pie")


# plot the named features 
# =============================================================================
# 
# Both plots show the density distribution of the values for the respective features,
# with the y-axis representing the density (probability) and the x-axis representing 
# the range of values. The shaded areas under the curves represent the density estimates.
# 
# From the plots, we can see that the "Amount" feature has a highly skewed distribution 
# with a long tail, indicating the presence of outliers or extreme values. On the other hand,
# the "Time" feature seems to have a relatively uniform distribution with no clear peaks 
# or patterns.
# 
# =============================================================================
f, axes = plt.subplots(1, 2, figsize=(18,4), sharex = True)

amount_value = dataset['Amount'].values # values
time_value = dataset['Time'].values # values

sns.kdeplot(amount_value, color="m", shade=True, ax=axes[0])
axes[0].set_title('Distribution of Amount')

sns.kdeplot(time_value, color="m", shade=True, ax=axes[1])
axes[1].set_title('Distribution of Time')

plt.show()

print("Average Amount in a Fraudulent Transaction : "+str(dataset[dataset["Class"] == 1]["Amount"].mean()))

print("Average Amount in a Valid Transaction : "+str(dataset[dataset["Class"] == 0]["Amount"].mean()))

print("Summary of the features - Amount" + "\n-------------------------------")
print(dataset["Amount"].describe())

# =============================================================================
# 
# The rest of the features don't have any physical interpretation and will be 
# seen through histograms. Here the values are subgrouped according to class 
# (valid or fraud)
# 
# =============================================================================

# Reorder the columns Amount, Time then the rest

data_plot = dataset.copy()

amount = data_plot['Amount']

data_plot.drop(labels=['Amount'], axis=1, inplace=True)

data_plot.insert(0, 'Amount', amount)

# Plot the distributions of the features
columns = data_plot.iloc[:,0:30].columns
plt.figure(figsize=(12,30*4))
grids = gridspec.GridSpec(30, 1)

for grid, index in enumerate(data_plot[columns]):
    ax = plt.subplot(grids[grid])
    sns.kdeplot(data=data_plot[data_plot.Class == 1], x=index, shade=True, alpha=0.5, label='Fraudulent')
    sns.kdeplot(data=data_plot[data_plot.Class == 0], x=index, shade=True, alpha=0.5, label='Genuine')
    ax.set_xlabel("")
    ax.set_title("Distribution of Column: " + str(index))
    ax.legend()

plt.show()

