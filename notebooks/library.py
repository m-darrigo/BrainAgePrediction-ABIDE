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















