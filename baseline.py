from data import *
import pandas as pd


def baseline_predictor(instance):
    return instance


def evaluate_model(model, instances, labels):
    pred = model(instances)
    mse = ((pred - labels) ** 2).mean(axis=ax)
    return mse


def run_baseline_model(no_days=5):
    instances, labels = get_baseline_data(no_days=no_days)
    split = int(0.8 * len(instances))
    test_data, test_labels = instances[split:], labels[split:]
    model = baseline_predictor
    evaluate_model(model, test_data, test_labels)


if __name__ == "__main__":
    run_baseline_model()
