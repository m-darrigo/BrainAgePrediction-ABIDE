"""
Regression Pipelines Module

This module provides functions for building and evaluating regression pipelines, specifically
focused on linear regression with L1 regularization (Lasso) and neural network regression
using Keras models.

Functions:
1. `linear_regressor_pipeline`:
   Conducts a linear regression pipeline with L1 regularization (Lasso) by scaling data,
   performing PCA for dimensionality reduction, and using Lasso regression.
   It optimizes the pipeline using grid search with cross-validation.

2. `n_layers_feed_forward`:
   Creates a feed-forward neural network with a specified number of layers, hidden units,
   and dropout rates. It returns a Keras Sequential model.

3. `neural_network_pipeline`:
   Conducts a pipeline for neural network regression using a specified Keras model.
   The pipeline includes scaling data, performing PCA for dimensionality reduction,
   and using the Keras model. It optimizes hyperparameters using randomized search
   with cross-validation.

Modules:
- `itertools.product`: Generates Cartesian product of input iterables.
- `typing`: Type hints for Python.

Dependencies:
- `numpy`: Numerical operations and random number generation.
- `pandas`: Data manipulation and analysis.
- `scikit-learn`: Machine learning library for preprocessing, modeling, and evaluation.
- `keras`: High-level neural networks API.
- `scikeras.wrappers.KerasRegressor`: Wrapper for Keras models compatible with scikit-learn.
- `numpy.typing`: Type hints for NumPy arrays.

Global Variables:
- `SEED`: Seed for the random number generator.
- `RNG`: Random number generator with seed `SEED`.
- `VALID_ENTRIES`: List of valid entries for the number of hidden units.
- `HIDDEN_LAYERS`: List of hidden layer configurations generated using `itertools.product`.
- `DROPOUTS`: List of dropout rates generated using `RNG.choice`.
- `PARAMS`: Dictionary containing hyperparameter settings for grid search.

Usage Example:
```python
import regression_pipelines as rp
import numpy as np

# Assuming 'df' and 'y' are your dataset and target variable
results_lasso = hp.linear_regressor_pipeline(df, y, verbose=1, seed=hp.SEED)
print(results_lasso)

# Create a KerasRegressor model for neural network regression
mlp = hp.n_layers_feed_forward(
    dropouts=[0.1, 0.0, 0.0, 0.0,],
    hiddens=(4, 4, 32, 32),
    nlayers=3,
    meta={"X_shape_":(0, 20)}
)
results_mlp = hp.neural_network_pipeline(mlp, df, y, niter=100, verbose=3, seed=hp.SEED)
print(results_mlp)
"""

from itertools import product
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import set_random_seed
from scikeras.wrappers import KerasRegressor
from numpy import typing as npt

# Random Number Generator
SEED = 20
RNG = np.random.RandomState(SEED)

# Set hidden layers and dropouts
VALID_ENTRIES = [4, 8, 16, 32, 64]
HIDDEN_LAYERS = list(product(VALID_ENTRIES, repeat=6))
DROPOUTS = list(
    RNG.choice([0.0, 0.1, 0.2, 0.3], p=[0.4, 0.2, 0.2, 0.2], size=(5_000, 6))
)

PARAMS = {
    "mlp__nlayers": np.random.randint(1, 7, size=22),
    "mlp__hiddens": HIDDEN_LAYERS,
    "mlp__dropouts": DROPOUTS,
    "mlp__optimizer__learning_rate": [0.0001, 0.001, 0.005],
}


def linear_regressor_pipeline(
    df: npt.NDArray[np.number],
    y: npt.NDArray[np.number],
    n_components: int = 20,
    thresh_overfit: float = 0.05,
    scoring: str = "r2",
    nsplits: int = 5,
    seed: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Conducts a linear regression pipeline using L1 regularization (Lasso).

    Parameters:
    -----------
    df : np.ndarray
        Input dataset containing features as a 2D array of numerical values.

    y : np.ndarray
        Target variable as a 1D array of numerical values.

    n_components : int, optional (default=20)
        Number of principal components to retain after dimensionality reduction.

    thresh_overfit : float, optional (default=0.05)
        Threshold for identifying overfitting, defined as the difference between
        mean training and test scores.

    scoring : str, optional (default="r2")
        Scoring metric used for the cross-validated grid search. Default is R-squared.

    nsplits : int, optional (default=5)
        Number of splits in the cross-validation strategy.

    seed : int, optional (default=None)
        Seed for random state to ensure reproducibility.

    **kwargs : Additional keyword arguments
        Additional parameters to be passed to the `GridSearchCV` function.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing cross-validation results for different hyperparameter
        settings. Only models without significant overfitting are included,
        sorted by mean test score in descending order.

    Raises:
    -------
    ValueError
        - If `n_components` is less than 1.
        - If `thresh_overfit` is negative.

    Notes:
    ------
    This function implements a linear regression pipeline with L1 regularization (Lasso).
    It scales the input data, performs principal component analysis (PCA) for dimensionality
    reduction, and uses Lasso regression as the final model. The pipeline is optimized
    using grid search with cross-validation to find the best hyperparameters.

    The results DataFrame includes information about the hyperparameter settings,
    training and test scores, and identifies models without significant overfitting.

    Example:
    --------
    ```python
    df = ...
    y = ...
    results = linear_regressor_pipeline(df, y, n_components=15, nsplits=10, seed=42)
    print(results)
    ```
    """
    # set a random state so the results are reproducible
    rng = np.random.RandomState(seed)

    # Input check
    if n_components < 1:
        raise ValueError("n_compontents must be higher than 1.")
    if thresh_overfit < 0:
        raise ValueError("thresh_overfit must be positive.")

    linear_regressor = Pipeline(
        [
            ("scaler", PowerTransformer()),
            (
                "principal_components",
                PCA(
                    n_components=n_components,
                    svd_solver="full",
                    random_state=rng,
                ),
            ),
            ("model", Lasso(random_state=rng)),
        ]
    )

    # L1 regularization
    param_distr = {"model__alpha": np.arange(1, 10)}

    search = GridSearchCV(
        linear_regressor,
        param_distr,
        return_train_score=True,
        scoring=scoring,
        cv=KFold(n_splits=nsplits, shuffle=True, random_state=rng),
        **kwargs,
    )

    search.fit(df, y)
    print("Done!")

    results_regressor = pd.DataFrame(search.cv_results_)
    no_overfit = (
        results_regressor["mean_train_score"] - results_regressor["mean_test_score"]
        < thresh_overfit
    )

    return results_regressor[no_overfit].sort_values("mean_test_score", ascending=False)


def n_layers_feed_forward(
    nlayers: int,
    hiddens: Iterable,
    dropouts: Iterable,
    meta: dict,  # Output KerasWrapper
):
    """
    Create a feed-forward neural network with a specified number of layers, hidden units,
    and dropout rates.

    Parameters:
    -----------
    nlayers : int
        Number of layers in the neural network.

    hiddens : Iterable
        Iterable containing the number of hidden units for each layer.

    dropouts : Iterable
        Iterable containing the dropout rates for each layer.

    meta : dict
        Metadata dictionary containing information about the input shape for the neural network.

    Returns:
    --------
    Sequential
        Keras Sequential model representing the specified feed-forward neural network.

    Raises:
    -------
    ValueError
        If `nlayers` is less than 1.

    Example:
    --------
    ```python
    nlayers = 3
    hiddens = [64, 128, 32]
    dropouts = [0.2, 0.4, 0.1]
    meta = {"X_shape_": (100,)}
    model = n_layers_feed_forward(nlayers, hiddens, dropouts, meta)
    ```

    """
    # set seed to have reproducible results
    set_random_seed(SEED)
    if nlayers < 1:
        raise ValueError("nlayers must be higher than 1")
    clf = Sequential()
    X_shape_ = (meta["X_shape_"][1],)

    clf.add(Dense(hiddens[0], activation="relu", input_shape=X_shape_))
    if dropouts[0] > 0:
        clf.add(Dropout(dropouts[0]))
    for i in range(1, nlayers):
        clf.add(Dense(hiddens[i], activation="relu"))
        if dropouts[i] > 0:
            clf.add(Dropout(dropouts[i]))
    clf.add(Dense(1))
    return clf


def neural_network_pipeline(
    mlp: KerasRegressor,  # Model
    df: npt.NDArray[np.number],
    y: npt.NDArray[np.number],
    n_components: int = 20,
    niter: int = 100,
    thresh_overfit: float = 0.05,
    scoring: str = "r2",
    nsplits: int = 5,
    seed: Optional[int] = None,
    **kwargs,
):
    """
    Conducts a pipeline for neural network regression using a specified Keras model.

    Parameters:
    -----------
    mlp : KerasRegressor
        KerasRegressor model to be used in the pipeline.

    df : np.ndarray
        Input features as a 2D array of numerical values.

    y : np.ndarray
        Target variable as a 1D array of numerical values.

    n_components : int, optional (default=20)
        Number of principal components to retain after dimensionality reduction.

    niter : int, optional (default=100)
        Number of iterations for randomized search during hyperparameter tuning.

    thresh_overfit : float, optional (default=0.05)
        Threshold for identifying overfitting, defined as the difference between
        mean training and test scores.

    scoring : str, optional (default="r2")
        Scoring metric used for the cross-validated randomized search. Default is R-squared.

    nsplits : int, optional (default=5)
        Number of splits in the cross-validation strategy.

    seed : int, optional (default=None)
        Seed for random state to ensure reproducibility.

    **kwargs : Additional keyword arguments
        Additional parameters to be passed to the `RandomizedSearchCV` function.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing cross-validation results for different hyperparameter
        settings. Only models without significant overfitting are included,
        sorted by mean test score in descending order.

    Example:
    --------
    ```python
    from sklearn.neural_network import MLPRegressor
    import hp_search

    # Assuming 'df' and 'y' are your dataset and target variable
    mlp_model = MLPRegressor()
    results = neural_network_pipeline(mlp_model, df, y, n_components=15, nsplits=10, seed=42)
    print(results)
    ```

    """

    rng = np.random.RandomState(seed)
    model = Pipeline(
        [
            ("scaler", PowerTransformer()),
            (
                "principal_components",
                PCA(
                    n_components=n_components,
                    svd_solver="full",
                    random_state=rng,
                ),
            ),
            ("mlp", mlp),
        ]
    )

    random_search = RandomizedSearchCV(
        model,
        PARAMS,
        refit=False,
        cv=KFold(nsplits, shuffle=True, random_state=rng),
        return_train_score=True,
        scoring=scoring,
        n_iter=niter,
        random_state=rng,
        **kwargs,
    )

    print("Starting grid search neural_network...")
    random_search.fit(df, y)
    print("Done!")

    df_results = pd.DataFrame(random_search.cv_results_)
    no_overfit = (
        df_results["mean_train_score"] - df_results["mean_test_score"] < thresh_overfit
    )

    return df_results[no_overfit].sort_values("rank_test_score")
