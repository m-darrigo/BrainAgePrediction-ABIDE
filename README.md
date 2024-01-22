[![Documentation Status](https://readthedocs.org/projects/brainageprediction-abide/badge/?version=latest)](https://brainageprediction-abide.readthedocs.io/en/latest/?badge=latest)

# BrainAgePrediction-ABIDE
This repository, managed by Marco D'Arrigo, exhibits my exam project for the Computing Methods for Experimental Physics and Data Analysis course. 
The project is centered around developing an algorithm capable of predicting brain age by analyzing features extracted from brain MRIs.

The dataset used is derived from the well-known ABIDE dataset, including subjects affected by Autism Spectrum Disorder (ASD) and healthy control subjects (CTR). 
The algorithms, crafted for this purpose, are equipped to perform Data Understanding & Preparation and implement regression models
## Dataset
The dataset contains 419 brain morphological features (volumes, thinckness, area, etc.) of brain parcels and global measure, derived for 915 male subjects of the [ABIDE dataset](http://fcon_1000.projects.nitrc.org/indi/abide/) ABIDE stands for Autism Brain Imaging Data Exchange.

The features are contained in a .csv file in the BrainAgePrediction/dataset folder.

For each subject, the following information are specified in the fists columns:
- subject ID
- age at scan
- sex
- Full intellectual quotient (FIQ)
- diagnostic group (DX_GROUP): 1 for Autism Spectrum Disorder (ASD), -1 for typical development (TD)
