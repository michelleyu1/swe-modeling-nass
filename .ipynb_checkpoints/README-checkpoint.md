# Modeling Snow Water Equivalent (SWE)

## Description
This repository contains code to train and validate regression and machine learning models for predicting SWE based on daily-scale predictions.

## Features
* Train regression, random forest, and neural network models.
* Validate model performance.
* Save and load the trained models.
* Apply the trained models for point-level prediction.

## Requirements
* python 3.12.2
* pytorch 2.1.2
* numpy 1.26.4
* pandas 2.2.1
* geopandas 0.14.3
* xarray 2024.2.0
* scikit-learn 1.4.1.post1
* statsmodels 0.14.1
* shapely 2.0.3
* richdem 2.3.0
* rasterio 1.3.9

## Usage
Here is an overview of what each file and directory in this project does:
* `data_scripts`: 
    * `eda.ipynb`: Simple exploratory data analysis.
* `evaluation_scripts`:
    * `evaluate.ipynb`: Compute performance metrics and visualize climatological model predictions.
* `modeling_scripts`:
    * `config.py`: Configurations for model training and prediction.
    * `data.py`: Functions to clean and organize dataframes.
    * `main.py`: The main entry point for the project. It orchestrates the overall execution of the pipeline.
    * `misc.py`: Miscellaneous functions.
    * `models.py`: Model definitions and training.
    * `time_series_dswe.py`: Generate continuous time series of SWE based on daily SWE predictions from trained models.
    * `train.py`: Initializes model training.
    
To initiate training, simply run `python main.py`. Once model training is complete, walking through `evaluate.ipynb` will compute and save metrics that evaluate the performance of the models.
    
## Data
* SNOwpack TELemetry Network (SNOTEL) Observational Data
* Parameter elevation Regression on Independent Slopes Model (PRISM) Gridded Climate Data
* Global 30 Arc-Second Elevation (GTOPO30) Digital Elevation Model
