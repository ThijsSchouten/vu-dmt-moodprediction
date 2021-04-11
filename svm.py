from sklearn import svm
from data import *


def fit_svm(instances, labels):
    model = svm.SVR(kernel="poly")
    model.fit(instances, labels)
    return regr


def evaluate_model(model, instances, labels):
    pred = model(instances)
    mse = ((pred - labels) ** 2).mean(axis=ax)
    return mse


def run_svm_model():
    instances, labels = get_aggregated_data()
    split = int(0.8 * len(instances))
    train_data, train_labels = instances[:split], labels[:split]
    test_data, test_labels = instances[split:], labels[split:]
    model = fit_svm(instances, labels)
