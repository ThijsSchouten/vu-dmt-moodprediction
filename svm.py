# %%
import random
import math

from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# from sklearn.model_selection import KFold
from data import *

from pickle import dump


def fit_svm(instances, labels, params):
    """
    Fits a SVR on instances&labels,
    using specified hyperparams.
    """
    model = svm.SVR(**params)
    model.fit(instances, labels)
    return model


def evaluate_model(model, instances, labels):
    """
    Evaluate the model by calculating 
    the MSE. 
    """
    prediction = model.predict(instances)

    prediction_inv = inverse_normalization(prediction)
    labels_inv = inverse_normalization(labels)
    mse = mean_squared_error(labels_inv, prediction_inv)

    return mse


def grid_search(instances, labels, cv=6):
    """
    Executes a SVR grid-search using
    on specified instances and labels
    crossvalidation. 

    Returns the best parameters as dict
    """
    grid = {
        "kernel": ("linear", "rbf", "sigmoid"),
        "C": [0.1, 1, 5, 10],
        "coef0": [0.01, 10, 0.5],
        "epsilon": [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10],
        "gamma": ("auto", "scale"),
    }

    model = svm.SVR()
    grid_search = GridSearchCV(
        model, grid, scoring="neg_mean_squared_error", n_jobs=-1, cv=cv, verbose=0
    )
    grid_search.fit(instances, labels)
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


def run_svm_model(train_data, train_labels, test_data, test_labels, gridsearch=False):

    # Use gridsearch to find the best parameters on the training data
    if gridsearch:
        best_params = grid_search(train_data, train_labels)
    else:
        best_params = {
            "C": 5,
            "coef0": 0.5,
            "epsilon": 0.01,
            "gamma": "auto",
            "kernel": "sigmoid",
        }

    # Fit SVR on training data
    model = fit_svm(train_data, train_labels, best_params)

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

            mse, best_params = run_svm_model(
                train_data, train_labels, test_data, test_labels, gridsearch
            )

            result = {
                "model": "svr",
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
        n=30, split=0.75, seed=42, no_days=range(1, 20), gridsearch=True, verbose=False
    )

    dump(results, open("results/SVM.pkl", "wb"))


# %%
