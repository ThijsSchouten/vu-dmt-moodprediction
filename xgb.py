# %%
import random
import math

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# from sklearn.model_selection import KFold
from data import *

from pickle import dump


def fit_xgb(instances, labels, params):
    """
    Fits a SVR on instances&labels,
    using specified hyperparams.
    """
    # model = svm.SVR(**params)
    model = xgb.XGBRegressor(objective="reg:squarederror", **params)
    model.fit(np.array(instances), np.array(labels))
    return model


def evaluate_model(model, instances, labels):
    """
    Evaluate the model by calculating
    the MSE.
    """
    prediction = model.predict(np.array(instances))

    prediction_inv = inverse_normalization(prediction)
    labels_inv = inverse_normalization(labels)
    mse = mean_squared_error(labels_inv, prediction_inv)

    return mse


def grid_search(instances, labels, cv=6, verbose=False):
    """
    Executes a SVR grid-search using
    on specified instances and labels
    crossvalidation.

    Returns the best parameters as dict
    """
    grid = {
        "max_depth": range(2, 15, 2),
        "n_estimators": range(60, 220, 40),
        "learning_rate": [0.1, 0.01, 0.05],
    }

    model = xgb.XGBRegressor(objective="reg:squarederror")

    grid_search = GridSearchCV(model, grid, n_jobs=-1, verbose=verbose, cv=cv,)

    grid_search.fit(
        np.array(instances),
        np.array(labels),
        eval_metric="rmse"
        # early_stopping_rounds=42,
        # verbose=True
    )

    best_params = grid_search.best_params_

    return best_params


def split_dataset(instances, labels, train_split=0.8):
    """
    Splits dataset into train/test
    instances and labels.
    """
    split = int(train_split * len(instances))
    train_data, train_labels = instances[:split], labels[:split]
    test_data, test_labels = instances[split:], labels[split:]

    return train_data, train_labels, test_data, test_labels


def shuffle_dataset(instances, labels, seed):
    """
    Shuffles instances and labels.
    """
    data = list(zip(instances, labels))

    if isinstance(seed, int):
        random.Random(seed).shuffle(data)
    else:
        random.Random().shuffle(data)

    instances, labels = zip(*data)

    return instances, labels


def run_xgb_model(
    train_data, train_labels, test_data, test_labels, gridsearch=False, verbose=False
):

    # Use gridsearch to find the best parameters on the training data
    if gridsearch:
        best_params = grid_search(train_data, train_labels, verbose=verbose)
    else:
        best_params = {
            "C": 5,
            "coef0": 0.5,
            "epsilon": 0.01,
            "gamma": "auto",
            "kernel": "sigmoid",
        }

    # Fit XGB on training data
    model = fit_xgb(train_data, train_labels, best_params)

    # Evaluate the model by predicting the test set
    mse = evaluate_model(model, test_data, test_labels)

    return mse, best_params


def run_n_times(
    n=5, split=0.75, seed=42, no_days=[5, 6], gridsearch=True, verbose=False
):
    """
    ..
    """
    # List to save results in
    results = []

    # Create random seeds
    random.seed(seed)
    seedlist = [math.floor(random.uniform(0, 100)) for i in range(n)]

    for days in no_days:
        # Get the data
        inst, lbl = get_aggregated_data(days)

        for idx, seed in enumerate(seedlist):
            instances, labels = shuffle_dataset(inst, lbl, seed)

            # Split into train/test split
            train_data, train_labels, test_data, test_labels = split_dataset(
                instances, labels, train_split=split
            )

            mse, best_params = run_xgb_model(
                train_data,
                train_labels,
                test_data,
                test_labels,
                gridsearch,
                verbose=verbose,
            )

            result = {
                "model": "xgb",
                "mse": mse,
                "aggregated_days": days,
                "iteration": idx + 1,
                "split": split,
                "gridsearch": gridsearch,
                "best_params": best_params,
            }

            if verbose:
                print(result)

            results.append(result)

    df = pd.DataFrame(results)

    return df


if __name__ == "__main__":

    results = run_n_times(
        n=30, split=0.75, seed=42, no_days=range(1, 20), gridsearch=True, verbose=True
    )

    print(results)
    dump(results, open("results/XGB.pkl", "wb"))


# %%
