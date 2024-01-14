""" This module provides some function to read, explore and preprocess an input
    dataframe cointaining a set of features.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PowerTransformer


def alldistr_img(df):
    """
    Plot the distributions of all features in a pandas DataFrame in a single image.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset containing features for which distributions will be plotted.

    Raises:
    -------
    ValueError
        If the input `df` is not a pandas DataFrame.

    Notes:
    ------
    This function uses seaborn's displot to visualize the distributions of all features
    in the input DataFrame. Each feature is represented in a separate subplot within
    a single image. The resulting image is saved as "imgs/all_distributions.png" in the
    current working directory.

    Example:
    --------
    ```python
    import pandas as pd
    from your_module import alldistr_img

    # Assuming 'df' is your dataset
    alldistr_img(df)
    ```
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
    """
    Visualizes top correlated features with a target variable using scatter plots.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing features.
    - y (pd.Series): Target variable to be correlated with features.
    - target_variable (str, optional): Column name of the target variable. Default is "AGE_AT_SCAN".
    - num_top_features (int, optional): Number of top features to consider. Default is 30.

    Returns:
    None

    Calculates absolute correlation between each feature in df and the target variable (y).
    Selects the top 'num_top_features' features based on correlation and generates scatter plots for
    each feature against the target variable. Scatter plots are arranged in a grid with 3 columns.

    Example:
    top_corr_relations(df, y)
    """
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
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to be analyzed.

    Returns:
    --------
    None

    Raises:
    -------
    ValueError
        If `df` is not a pandas DataFrame.

    Notes:
    ------
    This function identifies columns with high variance based on kurtosis, applies Standard Scaler
    and PowerTransformer to the selected columns, and visualizes the scaled distributions using
    Seaborn's displot.

    The top 20 columns with the highest kurtosis values are chosen for visualization. Two scalers,
    Standard Scaler and PowerTransformer, are applied to create two sets of scaled distributions.
    Each set is displayed in a separate subplot with a title indicating the scaler used.

    Example:
    --------
    ```python
    import pandas as pd
    from library import scaled_distributions

    # Assuming 'df' is your DataFrame
    scaled_distributions(df)
    ```

    """
    # dtype check
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    # Identify columns with high variance based on kurtosis
    high_variance_col = df.kurtosis().sort_values(ascending=False).head(20).index

    # Define scalers
    scalers = [StandardScaler(), PowerTransformer()]

    # Define scaler names for plot titles
    names = ["Standard Scaler", "PowerTransformer"]

    # Iterate over scalers
    for scaler, name in zip(scalers, names):
        # Apply scaler and create a DataFrame with scaled values
        scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

        # Visualize scaled distributions using Seaborn's displot
        sns.displot(
        data=scaled_df[high_variance_col].melt(),
        x="value",
        col="variable",
        facet_kws={"sharey": False, "sharex": False},
        common_bins=False,
        col_wrap=5)

        plt.title(name)

    pass

def pca_variance(df):
    """
    Visualizes the explained variance of Principal Components Analysis (PCA) on scaled data.

    This function takes a DataFrame, applies two scalers (Standard Scaler and PowerTransformer),
    performs PCA, and visualizes the explained variance.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to be analyzed.

    Returns:
    --------
    None

    Raises:
    -------
    ValueError
        If `df` is not a pandas DataFrame.

    Notes:
    ------
    This function applies Standard Scaler and PowerTransformer to the DataFrame,
    performs PCA on the scaled data, and visualizes the explained variance.

    The top 20 columns with the highest kurtosis values are chosen for PCA analysis.
    The cumulative explained variance is plotted against the number of principal components.
    A vertical dashed line indicates the top 20 components, providing insight into the
    retained variance with a specific number of principal components.

    Example:
    --------
    ```python
    import pandas as pd
    from library import pca_variance

    # Assuming 'df' is your DataFrame
    pca_variance(df)
    ```

    """
    # dtype check
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")

    # Define scalers
    scalers = [StandardScaler(), PowerTransformer()]

    # Iterate over scalers
    for scaler in scalers:
        # Apply scaler and create a DataFrame with scaled values
        scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Perform PCA on the scaled data
    pca = PCA().fit(scaled_df)

    # Plot the explained variance and cumulative explained variance
    plt.vlines(20, 0, 1, ls="--", color="black", label=r"EV top 20 components")
    plt.plot(np.concatenate([[0.0], pca.explained_variance_ratio_.cumsum()]), label="Cumsum EV")
    plt.legend(loc="lower right")

    # Customize plot
    plt.xticks(range(0, 400, 100))
    plt.xlim(-20, df.shape[1])
    plt.xlabel("Components")
    plt.ylabel("Explained variance")

    pass
