# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import pickle

""" Encoder non-numerical values into factors
Later we can think about more approaches to input category encoding
"""

# Hardcoded data files
DATA_FILEPATH = "./data/interim/data_encoded.csv"
OUT_FILEPATH = "./data/processed/data_predicted.csv"

""" Runs data cleaning scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    
    This function severely lacks any type of error checking. It was built
    with a specific dataset in mind but with minor tweaks it should be
    fairly straightforward to extend these methods to any other dataset.
"""
logger = logging.getLogger(__name__)
logger.info('Train randomForest model...')

# Read data using pandas as simple as possible
X = pd.read_csv(DATA_FILEPATH)
X.drop('Unnamed: 0', axis=1, inplace=True)

# Subset for speed
X = X.iloc[1:1000,:]

# Assume column labels in 'status_group'
y = X.pop('status_group')
y = y.map({'functional':0, 'functional needs repair':1, 'non functional':2})

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3, 
                                                    random_state=420)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)
forest.fit(X_train, y_train)
forest.predict(X_test)
cross_val_score(forest, X_train, y_train, cv=3)

# Get feature importances
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, X.columns[f], importances[indices[f]]))



# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(X.columns, importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(X.columns, X.columns)
plt.xlim([-1, X.shape[1]])
plt.show()
# 
# 
# #try with logistic regression to establish a baseline
# model = LogisticRegression()
# model.fit(X,y)
# print(X.head())
# cross_val_score(model, X, y, cv=10)
# 
# #
# #define and fit model - i found 1000 estimators gave the best score
# model = RandomForestClassifier(n_estimators=1000)
# model.fit(X_train,y_train)
# #warning this takes forever
# print("Cross validation...")
# cross_val_score(model, X_train, y_train, 'accuracy', cv=10)
# #warning this takes forever
# cross_val_score(forest, X, y, 'accuracy', cv=3)
# 
# # save the model to disk
# filename = './models/exploratory/feature_importance_tree.sav'
# # pickle.dump(forest, open(filename, 'wb'))
# 


