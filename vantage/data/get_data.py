import pandas as pd
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    
    # Read data from s3 amazon instance
    data = pd.read_csv('https://s3.amazonaws.com/drivendata/data/7/public/4910797b-ee55-40a7-8668-10efd5c1b960.csv', encoding = 'iso-8859-1')
    data_outfile = os.path.join(project_dir, 'data/raw', 'water_pump_set.csv')
    data.to_csv(data_outfile, index=False)
    print(f"Data file downloaded to %s" % data_outfile)
    
    labels = pd.read_csv('https://s3.amazonaws.com/drivendata/data/7/public/0bf8bc6e-30d0-4c50-956a-603fc693d966.csv', encoding = 'iso-8859-1')
    labels_outfile = os.path.join(project_dir, 'data/raw', 'water_pump_labels.csv')
    labels.to_csv(labels_outfile, index=False)
    print(f"Label file downloaded to %s" % labels_outfile)
    
    # A .docx was provided with feature descriptions. This was formatted as a csv file.
    # TODO MAKE SURE THIS FILE IS MADE WITHIN THIS SCRIPT!
    # ORIGINAL WAS A .docx FILE
    features = pd.DataFrame([\
        ['amount_tsh', 'Total static head (amount water available to waterpoint)'],\
        ['date_recorded', 'The date the row was entered'],\
        ['funder', 'Who funded the well'],\
        ['gps_height', 'Altitude of the well'],\
        ['installer', 'Organization that installed the well'],\
        ['longitude', 'GPS coordinate '],\
        ['latitude', 'GPS coordinate'],\
        ['wpt_name', 'Name of the waterpoint if there is one'],\
        ['num_private', ' '],\
        ['basin', 'Geographic water basin'],\
        ['subvillage', 'Geographic location'],\
        ['region', 'Geographic location'],\
        ['region_code', 'Geographic location (coded)'],\
        ['district_code', 'Geographic location (coded)'],\
        ['lga', 'Geographic location'],\
        ['ward', 'Geographic location'],\
        ['population', 'Population around the well'],\
        ['public_meeting', 'True/False'],\
        ['recorded_by', 'Group entering this row of data'],\
        ['scheme_management', 'Who operates the waterpoint'],\
        ['scheme_name', 'Who operates the waterpoint'],\
        ['permit', 'If the waterpoint is permitted'],\
        ['construction_year', 'Year the waterpoint was constructed'],\
        ['extraction_type', 'The kind of extraction the waterpoint uses'],\
        ['extraction_type_group', 'The kind of extraction the waterpoint uses'],\
        ['extraction_type_class', 'The kind of extraction the waterpoint uses'],\
        ['management', 'How the waterpoint is managed'],\
        ['management_group', 'How the waterpoint is managed'],\
        ['payment', 'What the water costs'],\
        ['payment_type', ' What the water costs'],\
        ['water_quality', 'The quality of the water'],\
        ['quality_group', 'The quality of the water'],\
        ['quantity', 'The quantity of water'],\
        ['quantity_group', 'The quantity of water'],\
        ['source', 'The source of the water'],\
        ['source_type', 'The source of the water'],\
        ['source_class', 'The source of the water'],\
        ['waterpoint_type', 'The kind of waterpoint'],\
        ['waterpoint_type_group', 'The kind of waterpoint']
    ], columns = ["Feature", "Description"])

    features_outfile = os.path.join(project_dir, 'data/raw', 'water_pump_features.csv')
    print(f"Feature description file created at %s" % labels_outfile)
    features.to_csv(features_outfile, index=False)



