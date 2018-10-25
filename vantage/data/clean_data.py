# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

""" 
    Runs data cleaning scripts to turn data from (../raw) into
    cleaned data ready to be analyzed (saved in ../interim).
    
    This function severely lacks any type of error checking. It was built
    with a specific dataset in mind but with minor tweaks it should be
    fairly straightforward to extend these methods to any other dataset.
    
    Code for data cleaning was partly taken from
    https://github.com/jdills26/Tanzania-water-table
    and 
    https://rstudio-pubs-static.s3.amazonaws.com/339668_006f4906390e41cea23b3b786cc0230a.html
    
    Steps in this function
        - Combine extraction_type, extraction_type_class and extraction_type_group
"""

# Hardcoded data files
DATA_FILEPATH = "./data/raw/water_pump_set.csv"
LABEL_FILEPATH = "./data/raw/water_pump_labels.csv"
OUT_FILEPATH = "./data/interim/data_cleaned.csv"

logger = logging.getLogger(__name__)
logger.info('Combining extraction columns...')

# Read data using pandas as simple as possible
df = pd.read_csv(DATA_FILEPATH)
ydat = pd.read_csv(LABEL_FILEPATH)

# Reformat date and merge with labels for intermediate analyses
df['date_recorded'] = pd.to_datetime(df['date_recorded'])
df = pd.merge(df, ydat, on='id') # Merge by id to guarantee correct labels

# Group low occuring counts together or into related groups
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Drop columns AFTER modifying other columns
# Define them here for clarity
columns_to_drop = [
    "recorded_by",
    "wpt_name",
    "scheme_name", 
    "scheme_management",
    "extraction_type_group", 
    "funder",
    "installer", 
    # "source_class",
    "source_type", # Maybe we could try source_type instead.
    "payment", 
    "waterpoint_type_group", 
    "management_group",
    "permit",
    "public_meeting",
    # "lga",
    # "subvillage",
    "amount_tsh",
    #"construction_year",
    "num_private",
    "population"
    ]

# Combine extraction type by manually combining low-occurring groups
df['extraction_type'][df.extraction_type.isin(['india mark ii','india mark iii'])] = 'india mark'
df['extraction_type'][df.extraction_type.isin(['cemo', 'climax'])] = 'motorpump'
df['extraction_type'][df.extraction_type.isin(['other - mkulima/shinyanga','other - play pump', 'walimi'])] = 'handpump'
df['extraction_type'][df.extraction_type.isin(['other - swn 81','swn 80'])] = 'swn'

# Logger not working as long as this is a script and not a Python module
# Will fix later!
logger.info('Remove unused features from data...')

#use lga column to create new feature: rural, urban, other
def map_lga(x):
    x=str(x)
    if x.find('Rural') >= 0: return 'rural'
    if x.find('Urban') >= 0: return 'urban'
    return 'other'
# Map rural, urban or other from the lga column
df['rural_urban'] = df.lga.apply(map_lga)

#use date time function in pandas
df['date_recorded'] = pd.to_datetime(df['date_recorded'])   # Reformat the date_recorded
median_construction_year = df.construction_year[df.construction_year!=0].median()
df['construction_year'] = df.construction_year.map(lambda x: median_construction_year if x == 0 else x) # Replace 0 with median
df['age'] = df.date_recorded.dt.year - df.construction_year # Track lifetime of waterpump in years

#remove extraction_type_group
df.drop(columns_to_drop, axis=1, inplace=True)

# Not logger.info but print() at least
print(f"Saving cleaned feature values to %s..." % OUT_FILEPATH)
df.to_csv(OUT_FILEPATH)

