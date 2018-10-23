# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('computing feature importance of numerical features...')
    
    # Read data using pandas as simple as possible
    # Expect labels in last column
    data = pd.read_csv(input_filepath)
    X = data.iloc[:, :-1]
    y = data.iloc[:,-1]
    
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)
    
    forest.fit(X, y)
    
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    
    indices = np.argsort(importances)[::-1]

    # save the model to disk
    filename = './models/exploratory/feature_importance_tree.sav'
    pickle.dump(forest, open(filename, 'wb'))

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X.shape[1]):
        print(f)
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(X.columns, importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(X.columns, X.columns)
    plt.xlim([-1, X.shape[1]])
    plt.show()
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
