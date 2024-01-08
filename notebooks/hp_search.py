from itertools import product
from typing import Iterable, Optional

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from keras.layers import Dense, Dropout
from keras.models import Sequential
from scikeras.wrappers import KerasRegressor
from numpy import typing as npt
from keras.utils import set_random_seed

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
    Pipeline for linear regressor
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
    ...
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

    return 1
