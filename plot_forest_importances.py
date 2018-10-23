"""
=========================================
Feature importances with forests of trees
=========================================

This examples shows the use of forests of trees to evaluate the importance of
features on an artificial classification task. The red bars are the feature
importances of the forest, along with their inter-trees variability.

As expected, the plot suggests that 3 features are informative, while the
remaining are not.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

from argparse import ArgumentParser

# Read arguments from command line
# https://stackoverflow.com/questions/1009860/how-to-read-process-command-line-arguments :)
parser = ArgumentParser()
parser.add_argument("-f", "--file", dest="filename", default="D:/GitHub/vantage-project/data/raw/water_pump_set.csv",
                    help="read input data from FILE", metavar="FILE")
parser.add_argument("-l", "--labels", dest="labelsfile", default="D:/GitHub/vantage-project/data/raw/water_pumps_labels.csv",
                    help="don't print status messages to stdout")

args = parser.parse_args()

# Read data using pandas as simple as possible
X = pd.read_csv(args.filename)
y = pd.read_csv(args.labelsfile)[:, 2]

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
