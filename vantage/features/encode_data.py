# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

import pickle

from sklearn.base import TransformerMixin
from feature_encoders import *

""" Encoder non-numerical values into factors
Later we can think about more approaches to input category encoding

Encoders largely taken from
https://rstudio-pubs-static.s3.amazonaws.com/339668_006f4906390e41cea23b3b786cc0230a.html

However, there are some variable changes and pipeline changes made.
For instance the amount_tsh is included in our model because I do have not personally
believe that we can impute this value using mean/median imputation. There is no way
to distinguish between its missing values and true zero values. 
"""

# Hardcoded data files
# TODO: make these editable as command line arguments and edit Makefile accordingly.
DATA_FILEPATH = "./data/interim/data_cleaned.csv"
OUT_FILEPATH = "./data/interim/data_encoded.csv"

# Read data using pandas as simple as possible
X = pd.read_csv(DATA_FILEPATH)
# Sometimes we have this artifact column, remove it.
if 'Unnamed: 0' in X.columns: X.drop('Unnamed: 0', axis=1, inplace=True)


# Assume column labels in 'status_group'
# Some ugly hard-coding :) We can define the label column either as the last column by default or at least as a variable/constant.
y = X.pop('status_group')

# -------------- SET MODEL VARIABLES ------------------- #
# define categorical variables to encode for model
# Changes to the model go here... for now..
categorical = [
  # 'funder',
 #'installer',
 #'scheme_management',
 'extraction_type',
 'management',
 'payment_type',
 'water_quality',
 'quantity']
 
#Columns with many levels.
high_levels_cat_columns = [
  'subvillage',
  'lga',
  'ward']

#Columns with less number of levels.
low_levels_cat_columns = [
        'basin',
        'region',
        'district_code',
        # 'public_meeting',
        # 'scheme_management',
        # 'permit',
        'extraction_type',
        # 'extraction_type_group',
        'extraction_type_class',
        'management',
        #'management_group',
        'payment_type',
        'water_quality',
        #'quality_group',
        'quantity_group',
        'source',
        #'source_type','source_class',
        'waterpoint_type',
        'rural_urban'
        #,'waterpoint_type_group'
        ]

#
unchanged_cols = ['age', 'construction_year']

# ----------Define Pipelines with Sklearn----------------#
# These pipelines only do the data transformation used for encoding the data
# Although I am very dissappointed that the 

cat_pipeline_high_level_freq_based = Pipeline([ \
                         ('selector',DataFrameSelector(high_levels_cat_columns)), \
                         ('cat_nulls', HandleCategoricalNulls()), \
                         ('FreqBasedCategoricalBinning', \
                          FreqBasedCategoricalBinning(buckets=20,apply=True)), \
                         ('CatMultiLabelTransformer',CatMultiLabelTransformer()) \
                         ])
                         
cat_pipeline_high_level_resp_based = Pipeline([ \
                         ('selector',DataFrameSelector(high_levels_cat_columns)), \
                         ('cat_nulls', HandleCategoricalNulls()), \
                         ('RespBasedCategoricalBinning', \
                          RespBasedCategoricalBinning(buckets=20,apply=True)), \
                         ('CatMultiLabelTransformer',CatMultiLabelTransformer()) \
                         ])

choose_high_level_cat_pipeline = Pipeline([ \
                                         ('ChooseCatPipelineType', ChooseCatPipelineType( \
                                         freq_pipeline = cat_pipeline_high_level_freq_based, \
                                         resp_pipeline = cat_pipeline_high_level_resp_based, \
                                         method = 'freq')) \
                                         ])
                                         
cat_pipeline_low_level = Pipeline([ \
                         ('selector',DataFrameSelector(low_levels_cat_columns)), \
                         ('cat_nulls', HandleCategoricalNulls()), \
                         ('CatMultiLabelTransformer',CatMultiLabelTransformer(apply=True)) \
                        ])

#Combine all categorical pipelines first:
full_categorical_pipeline = FeatureUnion(transformer_list = [ \
                                         ("choose_high_level_cat_pipeline",\
                                          choose_high_level_cat_pipeline), \
                                         ("cat_pipeline_low_level",\
                                         cat_pipeline_low_level) \
                                         ])       
#Lat Long pipelines
lat_long_prep_pipeline = Pipeline([ \
                         ('selector',DataFrameSelector(['lga',\
                                                        'region',\
                                                        'ward'])), \
                         ('cat_nulls', HandleCategoricalNulls()) \
                         ])

lat_long_selector = Pipeline([ \
                         ('selector',DataFrameSelector(['longitude','latitude']))])

lat_long_transformer = Pipeline([ ('lat_long_prep',\
                                   FeatureUnion(transformer_list = [ \
                                    ("lat_long_prep_pipeline",\
                                     lat_long_prep_pipeline), \
                                    ("lat_long_selector",\
                                     lat_long_selector)])) \
                                    ,("Numpy2DFTransformer", \
                                      Numpy2DFTransformer(['lga','region',\
                                                           'ward','longitude',\
                                                           'latitude'])) \
                                    ,('LatitudeLongitudeProcess', \
                                      LatitudeLongitudeProcess(strategy="custom")) \
                                ])
#lat_long_transformer always return the 
#data in latitude, longitude order
# 
# #gps_height pipelines.
# #This is dependent on the lat_long pipeline
# gps_height_transformer = Pipeline([('gps_height_prep',\
#                                     FeatureUnion(transformer_list=[('lat_long_transformer', \
#                                                                    lat_long_transformer), \
#                                    ('gps_selector',DataFrameSelector(['gps_height']))]))
#                                    ,("Numpy2DFTransformer", \
#                                       Numpy2DFTransformer(['latitude','longitude','gps_height']))
#                                    # ,('GPSHeightTransformer',GPSHeightTransformer(method='median'))
#                                    #,('GPSHeightTransformer',GPSHeightTransformer(method='custom'))
#                                   ])
# 
# 

#population pipelines.
#This is dependent on the lat_long pipeline

population_transformer = Pipeline([('population_prep',\
                                    FeatureUnion(transformer_list=[('lat_long_transformer', \
                                                                   lat_long_transformer), \
                                   ('population_selector',DataFrameSelector(['population']))]))
                                   ,("Numpy2DFTransformer", \
                                      Numpy2DFTransformer(['latitude','longitude','population']))
                                   # ,('PopulationTransformer',PopulationTransformer(method = 'ignore'))
                                   #,('PopulationTransformer',PopulationTransformer(method = 'custom')) #NOT worth
                                   # ,('PopulationTransformer',PopulationTransformer(method = 'median')) #NOT Worth
                                  ])
                                  
unchanged_column_selector = Pipeline([ \
                         ('selector',DataFrameSelector(unchanged_cols))])

full_numeric_transformations = Pipeline([('all_numeric_transformations', \
                                        FeatureUnion(transformer_list = [\
                                        ('lat_long_transformer',lat_long_transformer), \
                                        # ('gps_height_transformer',gps_height_transformer), \
                                        # ('population_transformer',population_transformer), \
                                        #('age_pipeline',age_pipeline) \
                                       ])) \
                                       ,('scaler',ScaleData())
                                        ])
                                        
all_transformations = Pipeline([ \
                                ('all_transformations', \
                                    FeatureUnion(transformer_list = \
                                                 [('unchanged_columns', unchanged_column_selector),
                                                 ('full_categorical_pipeline',\
                                                   full_categorical_pipeline), \
                                                  ('full_numeric_transformations',\
                                                   full_numeric_transformations)])) \
                                 ])        
  
df_encoded = all_transformations.fit_transform(X, y)

all_columns = unchanged_cols + \
              choose_high_level_cat_pipeline.named_steps['ChooseCatPipelineType'].column_names + \
              cat_pipeline_low_level.named_steps['CatMultiLabelTransformer'].column_names + \
              lat_long_transformer.named_steps['LatitudeLongitudeProcess'].column_names

df_encoded_pd = pd.DataFrame(df_encoded, columns=all_columns)
df_encoded_pd['status_group'] = y
print(df_encoded_pd.head())
df_encoded_pd.to_csv(OUT_FILEPATH)
