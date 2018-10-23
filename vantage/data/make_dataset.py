# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier


@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('label_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(data_filepath, label_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # Read data using pandas as simple as possible
    df = pd.read_csv(data_filepath)
    ydat = pd.read_csv(label_filepath)
    y = ydat['status_group'] # Big assumption that labels are in the second column of the labels dataframe (NOT USUALLY TRUE)
    
    # Reformat date and merge with labels for intermediate analyses
    df['date_recorded'] = pd.to_datetime(df['date_recorded'])
    df = pd.merge(df, ydat, on='id')
    
    # Write to csv using default options so that we can usually also read
    # it again with default options
    # df.to_csv("./data/interim/water_pumps_sets_with_labels.csv")
    
    #
    y = df['status_group']
    X = df.select_dtypes(['int64', 'float64'])
    
    # Write data of numerical feature values
    # Include labels in the last column
    df_num = pd.merge(X, ydat,on='id')
    # df_num_file = "./data/interim/water_pumps_sets_numeric.csv"
    print("Saving numerical feature values to %s..." % output_filepath)
    df_num.to_csv(output_filepath)
    
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
