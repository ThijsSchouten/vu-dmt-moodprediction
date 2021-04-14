import random

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from data import *


def fit_xgb(instances, labels, params):
    """
    Fits a SVR on instances&labels,
    using specified hyperparams.
    """
    # model = svm.SVR(**params)
    model = xgb.XGBRegressor(objective='reg:squarederror', **params)
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


def grid_search(instances, labels, cv=6):
    """
    Executes a SVR grid-search using
    on specified instances and labels
    crossvalidation.

    Returns the best parameters as dict
    """
    grid = {
        'max_depth': range(2, 10, 1),
        'n_estimators': range(60, 220, 40),
        'learning_rate': [0.1, 0.01, 0.05]
    }

    model = xgb.XGBRegressor(objective='reg:squarederror')

    grid_search = GridSearchCV(model,
                               grid,
                               n_jobs=-1,
                               verbose=True,
                               cv=2)  # ,
    #                            #    cv=cv,
    #                            #    verbose=True
    #                            )

    grid_search.fit(np.array(instances), np.array(labels),
                    eval_metric='rmse', verbose=True)

    best_params = grid_search.best_params_

    print(best_params)

    return "best_params"


def split_dataset(instances, labels, train_split=.8):
    """
    Splits dataset into train/test
    instances and labels.
    """
    split = int(train_split * len(instances))
    train_data, train_labels = instances[:split], labels[:split]
    test_data, test_labels = instances[split:], labels[split:]

    return train_data, train_labels, test_data, test_labels


def shuffle_dataset(instances, labels, seed=4):
    """
    Shuffles instances and labels in
    the same manner.
    """
    data = list(zip(instances, labels))
    random.Random(seed).shuffle(data)
    instances, labels = zip(*data)

    return instances, labels


def run_xgb_model(gridsearchCV=False, no_days=5, random_state=42):
    # Get the data and shuffle
    instances, labels = shuffle_dataset(
        *get_aggregated_data(no_days), random_state)

    # Split into train/test split
    train_data, train_labels, test_data, test_labels = split_dataset(
        instances, labels, train_split=.7)

    # Use gridsearch to find the best parameters on the training data
    best_params = {'learning_rate': 0.05,
                   'max_depth': 2, 'n_estimators': 60}

    if gridsearchCV:
        best_params = grid_search(train_data, train_labels)

    # Fit XGB on training data
    model = fit_xgb(train_data, train_labels, best_params)

    # Evaluate the model by predicting the test set
    mse = evaluate_model(model, test_data, test_labels)

    print(mse)


if __name__ == "__main__":
    for x in range(1, 21):
        print(x, end="day avg MSE: ")
        run_xgb_model(gridsearchCV=False, no_days=x)
