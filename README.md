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
    
Note: you may have to create the `data/` repositories first, you need

  - data/raw
  - data/interim
  - data/external
  - data/processed

    
To then create processed data and run all rules in the Makefile, activate your environment
and then call

    make all

This will,

  - check your conda environment
  - get the data from S3 server
  - clean the data and encode the data for model training
  - Create a report website under `reports/Report_site/index.html' with some exploratory analysis
    * Really this was more of a proof of concept.
  
Since it is unfinished it is not included in the Makefile, but...
  - With your activated environment you can run `python vantage/models/train_model.py
    * This will train an extremely basic random forest on 70% of the data
    * Output is not written anywhere
    * Output presentation in Rmarkdown was removed.
  
    

To run the computational data transformation steps:
    
    make finaldataset

This will only create a cleaned up subset of the data and an encoded version of the data to use
in Python for classification modelling using one-hot encoding of included categorical data.

### Report

To create a small report 'website', run:
  
    make report
    
The output can be found under `reports/Report_site/index.html!
This is created using Rmarkdown and the `rmarkdown::render_site()` function.
You can use this, instead of jupyter notebooks, to do your data analysis and exploratory analysis
in multiple Rmarkdown files and then knit them together as a website. I think a website output for
larger analyses is well-suited if there are many technical steps that may not be relevant to every reader.

### Rstudio

You can use RStudio and the `reticulate` package, enabling you to run Python code within
Rmarkdown documents or R scripts. To make RStudio use the Conda environment, make sure to
install Rstudio version>1.2 OUTSIDE your conda environment.

You can download it here, https://www.rstudio.com/products/rstudio/download/preview/

Using `conda install rstudio` will give you an older version of RStudio which
includes less features for using Python code within R.

You can then run

    source activate vantage-project
    rstudio 
    
This will open rstudio using your conda environment.
Verify this by opening RStudio and running

    library(reticulate)
    py_config()

You should see something as follows

    python:         D:\Anaconda\envs\vantage-project\python.exe
    libpython:      D:/Anaconda/envs/vantage-project/python36.dll
    pythonhome:     D:\Anaconda\envs\vantage-project
    version:        3.6.6 |Anaconda, Inc.| (default, Jun 28 2018, 11:27:44) [MSC v.1900 64 bit (AMD64)]
    Architecture:   64bit
    numpy:          D:\Anaconda\envs\vantage-project\lib\site-packages\numpy
    numpy_version:  1.15.2

Tanzania water pump maintenance prediction
==============================

In this repository you can find notebooks under `/notebooks/dev'.
These show some exploratory analysis using Jupyter notebooks.

Sadly, scripts are scattered and older scripts were removed prior to pushing this last version.

A more detailed description of this repository may follow.

