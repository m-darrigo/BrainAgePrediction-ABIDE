""" This module provides some function to read, explore and preprocess an input
    dataframe cointaining a set of features.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PowerTransformer


def alldistr_img(df):
    """
    function that takes a dataset and plots the distributions of all features into a single image.
    """

    # dtype check
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    sns.displot(
        data=df.melt(),
        x="value",
        col="variable",
        facet_kws={"sharey": False, "sharex": False},
        common_bins=False,
        col_wrap=5,
    )

    plt.savefig("imgs/all_distributions.png")
    plt.close()

    pass

def corr_distr(df_corr):
    '''
    Function that calculates the correlation and then generates a distribution.
    '''
    np.fill_diagonal(df_corr.values, 0)
    bivariate_corr = df_corr.unstack().rename("correlation_coefficient").to_frame()
    sns.histplot(bivariate_corr)
    plt.xlabel("pearson correlation coefficient")

    pass

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

    pass

def scaled_distributions(df):
    """
    Visualizes scaled distributions of columns with high variance.

    This function takes a DataFrame and applies two scalers (Standard Scaler and PowerTransformer)
    to columns with high variance. Subsequently, it visualizes the scaled distributions using
    Seaborn's displot.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to be analyzed.

    Returns:
    None
    """
    high_variance_col = df.kurtosis().sort_values(ascending=False).head(20).index
    scalers = [StandardScaler(), PowerTransformer()]

    names = ["Standard Scaler", "PowerTransformer"]

    for scaler, name in zip(scalers, names):
        scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        g = sns.displot(
            data=scaled_df[high_variance_col].melt(),
            x="value",
            col="variable",
            facet_kws={"sharey":False, "sharex":False},
            common_bins=False,
            col_wrap=5
        )
        plt.title(name)

    pass

def pca_variance(df):
    """
    Visualizes the explained variance of Principal Components Analysis (PCA) on scaled data.

    This function takes a DataFrame, applies two scalers (Standard Scaler and PowerTransformer),
    performs PCA, and visualizes the explained variance.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data to be analyzed.

    Returns:
    None
    """
    high_variance_col = df.kurtosis().sort_values(ascending=False).head(20).index
    scalers = [StandardScaler(), PowerTransformer()]

    names = ["Standard Scaler", "PowerTransformer"]

    for scaler, name in zip(scalers, names):
        scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    pca = PCA().fit(scaled_df)

    plt.vlines(20, 0, 1, ls="--", color="black", label=r"EV top 20 components")
    plt.plot(np.concatenate([[0.0], pca.explained_variance_ratio_.cumsum()]), label="Cumsum EV")
    plt.legend(loc="lower right")

    plt.xticks(range(0 , 400, 100))
    plt.xlim(-20, df.shape[1])
    plt.xlabel("Components")
    plt.ylabel("Explained variance")

    pass







