[![Documentation Status](https://readthedocs.org/projects/brainageprediction-abide/badge/?version=latest)](https://brainageprediction-abide.readthedocs.io/en/latest/?badge=latest)

# BrainAgePrediction-ABIDE
This repository, managed by Marco D'Arrigo, exhibits my exam project for the Computing Methods for Experimental Physics and Data Analysis course. 
The project is centered around developing an algorithm capable of predicting brain age by analyzing features extracted from brain MRIs.

The dataset used is derived from the well-known ABIDE dataset, including subjects affected by Autism Spectrum Disorder (ASD) and healthy control subjects (CTR). 
The algorithms, crafted for this purpose, are equipped to perform Data Understanding & Preparation and implement regression models
## Dataset
The dataset contains 419 brain morphological features (volumes, thinckness, area, etc.) of brain parcels and global measure, derived for 915 male subjects of the [ABIDE dataset](http://fcon_1000.projects.nitrc.org/indi/abide/) ABIDE stands for Autism Brain Imaging Data Exchange.

# Regression Modeling for Age Prediction from Structural Brain Features

This project aims to design and compare regression models for predicting the age of healthy subjects based on structural brain features. The models include both traditional machine learning and deep learning approaches.

## General Structure

- **Data Directory:** Contains the dataset.
  
- **Main Project Directory (brain\_age\_predictor):**
  - **libs:** Includes two libraries, one with functions for preprocessing and another with pipelines for a linear regressor and an MLP.
  - **notebooks:** Contains two Jupyter notebooks, one for preprocessing and the other for regression, using the library functions.
  - **results:** Includes two .csv files with the best results of the regressors.
  - **imgs:** A directory for saving images.
  
- **Tests Directory (tests):** Includes simple tests for some functions.

- **Requirements File (requirements.txt):** Lists project dependencies.

- **Documentation (docs):**
  - Produced using Sphinx.
  - Accessible on Read the Docs.
  - Can be generated locally.
  - Utilizes `intersphinx_mapping` for navigation.

## Scripts and Notebooks

- **preprocessing\_lib.py:** Provides functions for data analysis and preprocessing.
  - **alldistr\_img:** Displays the distributions of all features in a single image.
  - **corr\_distr:** Calculates and displays the distribution of correlations.
  - **top\_corr\_relations:** Shows relationships between the most correlated features with the target variable.
  - **scaled\_distributions:** Displays normalized distributions with Standard Scaler and PowerTransformer for columns with high variance (for handling outliers).
  - **pca\_variance:** Displays the explained variance of Principal Component Analysis (PCA) on normalized data.
  
- **preprocessing.ipynb:** Jupyter notebook for data analysis and preparation.

- **regression\_pipelines.py:** Provides functions for the construction and evaluation of regression pipelines.
  - **linear\_regressor\_pipeline:** Conducts a linear regression pipeline with L1 regularization (Lasso) and optimization through a grid search with cross-validation.
  - **n\_layers\_feed\_forward:** Creates a feed-forward neural network.
  - **neural\_network\_pipeline:** Conducts a pipeline for regression with an MLP (Multi Layer Perceptron) and optimizes hyperparameters through a random search with cross-validation.
  
- **regression.ipynb:** Jupyter notebook that integrates the entire flow, applying the linear regression pipeline and training the neural network.

## Performance Results

- Linear Regressor: Achieved an $R^2$ score of 0.66 with a standard deviation of 0.04.

- MLP: Achieved an $R^2$ score of 0.71 with a standard deviation of 0.02.
