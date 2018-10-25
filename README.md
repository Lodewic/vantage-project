vantage-project
==============================

A data science case for Vantage AI

<p><small>Project based on the <a target="_blank" href="https://github.com/BigDataRepublic/cookiecutter-data-science">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

Note: This data science project is unfinished so the predictive modelling and 
classification steps are left out. These scripts' results were not satisfying nor
documented.

Presented here are the steps that WERE made. This includes,

  - Conda environment setup
  - Use of Makefile to run parts of the analysis
  - Creation of report website using R
    * Excluding modelling results as these were unfinished
    * Modelling scripts would take too much time to streamline

## Getting started:

One should be up and running as follows:

    make create_environment
    source activate vantage-project
    make requirements
    
The data for this project was already hosted on Amazon S3, to sync this data use

    make sync_data_from_s3
    
Note: you may have to create the `data/` repositories first, you need

  - data/raw
  - data/interim
  - data/external
  - data/processed

To run the computational data transformation steps:
    
    make finaldataset



This will only create a cleaned up subset of the data and an encoded version of the data to use
in Python for classification modelling using one-hot encoding of included categorical data.

To create a small report 'website', run:
  
    make report
    
The output can be found under `reports/Report_site/index.html!
This is created using Rmarkdown and the `rmarkdown::render_site()` function.
You can use this, instead of jupyter notebooks, to do your data analysis and exploratory analysis
in multiple Rmarkdown files and then knit them together as a website. I think a website output for
larger analyses is well-suited if there are many technical steps that may not be relevant to every reader.


Tanzania water pump maintenance prediction
==============================

In this repository you can find notebooks under `/notebooks/dev'.
These show some exploratory analysis using Jupyter notebooks.

Sadly, scripts are scattered and older scripts were removed prior to pushing this last version.

A more detailed description of this repository may follow.

