# %%
import random
import math
import time

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
        "coef0": [0.01, 0.5],
        "epsilon": [0.0001, 0.001, 0.01, 0.1, 1],
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


def run_svm_model(train_data, train_labels, test_data, test_labels, params=False):

    # Use gridsearch to find the best parameters on the training data
    if not params:
        params = grid_search(train_data, train_labels)

    # Fit SVR on training data
    model = fit_svm(train_data, train_labels, params)

    # Evaluate the model by predicting the test set
    mse = evaluate_model(model, test_data, test_labels)

    return mse, params


def run_n_times(
    n=5,
    split=0.75,
    seed=42,
    no_days=[5, 6],
    gridsearch=True,
    verbose=False,
    features_to_exclude={},
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
        inst, lbl = get_aggregated_data(days, features_to_exclude)
        best_params = False

        for idx, seed in enumerate(seedlist):
            instances, labels = shuffle_dataset(inst, lbl, seed)

            # Split into train/test split
            train_data, train_labels, test_data, test_labels = split_dataset(
                instances, labels, train_split=split
            )

            mse, best_params = run_svm_model(
                train_data, train_labels, test_data, test_labels, params=best_params
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
    score = df.mse.mean()

    return df, score


def save_model(model, path=False):
    assert isinstance(path, str)

    dump(model, open(path, "wb"))


def save_or_append(df, path=False):
    assert isinstance(path, str)

    try:
        saved_df = load(open(path, "rb"))
        df = pd.concat([df, saved_df])
    except:
        print("File not found.")

    dump(df, open(path, "wb"))


def find_best_params():
    filepath = "results/SVM_gridsearch_and_features_v2.pkl"

    selected = {"mood": "mean"}
    featurelist = get_featurelist()
    featurelist.pop("mood")

    # Keep track of the scores
    for i in range(1, 7):
        print(f"\n[ {i+1} FEATURES ] ")
        highscore, best_feature = 10, ""

        # For every item in the featurelist
        for key in featurelist:
            start = time.time()

            # Create a dict excluding this key
            current = featurelist.copy()
            current.pop(key)

            # Run the analysis, excluding all items in this dict
            # (thus including the current key)
            df, score = run_n_times(
                n=1,
                split=0.75,
                seed=42,
                no_days=range(1, 15),
                gridsearch=True,
                verbose=False,
                features_to_exclude=current,
            )

            # Add the featurelist as column to DF
            feature_list = list(selected)
            feature_list.append(key)
            df["features"] = ", ".join(feature_list)
            df["feature_count"] = len(feature_list)

            # Save to file
            save_or_append(df, filepath)

            if highscore > score:
                highscore = score
                best_feature = key

            end = time.time()
            print(
                f"Time: {round(end-start)}s | MSE: {round(score, 5)} | Features: {feature_list}"
            )

        # Pop best feature from the list,
        # add to the base features
        popped = featurelist.pop(best_feature)
        selected[best_feature] = popped
        print(f"-- Best score: {highscore} from {best_feature} \n")


def run_once():

    # to_keep = [
    #     "mood",
    #     "activity_night",
    #     # "screen_night",
    #     "circumplex.arousal",
    #     "circumplex.valence",
    #     # "appCat.social",
    # ]

    to_keep = get_featurelist()
    to_keep = to_keep.keys()

    exclude = get_featurelist()
    for feature in to_keep:
        exclude.pop(feature)

    # Run the analysis, excluding all items in this dict
    # (thus including the current key)
    df, score = run_n_times(
        n=100,
        split=0.75,
        seed=21,
        no_days=range(1, 21),
        gridsearch=True,
        verbose=False,
        features_to_exclude=exclude,
    )

    print(score)

    # Add the featurelist as column to DF
    df["features"] = "_".join(to_keep)

    # # Save to file
    # save_or_append(
    #     df, f"results/MSE:{round(score,4)}__FEATURES_{','.join(to_keep)}.pkl"
    # )

    # Save to file
    save_or_append(df, f"results/MSE:{round(score,4)}_RANGE_1-20_FEATURES_ALL_30x.pkl")


if __name__ == "__main__":
    # find_best_params()  # %%
    run_once()

# %%
