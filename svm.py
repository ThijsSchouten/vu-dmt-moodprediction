# %%
import random

from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from data import *


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
    # prediction = [x * 10 for x in prediction]
    # labels = [x * 10 for x in labels]
    return mean_squared_error(labels, prediction)


def grid_search(instances, labels, cv=6):
    """
    Executes a SVR grid-search using
    on specified instances and labels
    crossvalidation. 

    Returns the best parameters as dict
    """
    grid = {'kernel': ('linear', 'rbf', 'sigmoid'),
            'C': [0.1, 1, 5, 10],
            'coef0': [0.01, 10, 0.5],
            'epsilon': [0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10],
            'gamma': ('auto', 'scale')}

    svr = svm.SVR()
    clf = GridSearchCV(svr, grid, cv=cv, verbose=42)
    clf.fit(instances, labels)
    best_params = clf.best_params_

    return best_params


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

def reverse_normalization():
    pass

def run_svm_model(grid_search=False, no_days=5):
    # Get the data and shuffle
    instances, labels = shuffle_dataset(*get_aggregated_data(no_days), 42)

    # Split into train/test split
    train_data, train_labels, test_data, test_labels = split_dataset(
        instances, labels, train_split=.7)

    # Use gridsearch to find the best parameters on the training data
    best_params = {'C': 5, 'coef0': 0.5, 'epsilon': 0.01,
                   'gamma': 'auto', 'kernel': 'sigmoid'}
    if grid_search:
        best_params = grid_search(train_data, train_labels)

    # Fit SVR on training data
    model = fit_svm(train_data, train_labels, best_params)

    # Evaluate the model by predicting the test set
    mse = evaluate_model(model, test_data, test_labels)

    print(mse)


if __name__ == "__main__":
    for x in range(1,20):
        print(x, end="day avg MSE: ")
        run_svm_model(grid_search=False, no_days=x)


# %%
