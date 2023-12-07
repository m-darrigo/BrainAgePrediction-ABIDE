""" This module provides some function to read, explore and preprocess an input
    dataframe cointaining a set of features.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA


def alldistr_img(df):
    '''
    function that takes a dataset and plots the distributions of all features into a single image.
    '''
    sns.displot(
        data=df.melt(),
        x="value",
        col="variable",
        facet_kws={"sharey":False, "sharex":False},
        common_bins=False,
        col_wrap=5
    )

    plt.savefig("imgs/all_distributions.png")
    plt.close()

    return 1

def corr_distr(df_corr):
    '''
    Function that calculates the correlation and then generates a distribution.
    '''
    np.fill_diagonal(df_corr.values, 0)
    bivariate_corr = df_corr.unstack().rename("correlation_coefficient").to_frame()
    sns.histplot(bivariate_corr)
    plt.xlabel("pearson correlation coefficient")

    return 1

def top_corr_relations(df, y, target_variable="AGE_AT_SCAN", num_top_features=30):
    '''
    Identifies the top 30 features in a DataFrame based on their absolute correlation with a target variable.
    
    Creates a new DataFrame by combining the selected features and the target variable in a long format.
    
    Generates scatter plots with regression lines for each selected feature against a specified variable.
    
    The resulting visualizations offer insights into the relationship between the top correlated features and
    the target variable across different subplots.
    
    The scatter plots illustrate how the values of these features vary concerning the specified variable.
    '''
    top_corr_features = (
        np.abs(df.corrwith(y))
        .sort_values(ascending=False)
        .head(num_top_features)
        .index
    )

    df_long = (
        pd.concat((df[top_corr_features], y), axis=1)
        .melt(id_vars=target_variable)
    )

    sns.lmplot(
        data=df_long,
        x="value",
        y=target_variable,
        col="variable",
        col_wrap=3,
        sharex=False,
        sharey=False,
        scatter_kws={"alpha": 0.3}
    )

    return 0












