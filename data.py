# Code for Data Mining Techniques project period 5 2020/2021 VU

# %% Import packages
from numpy.core.arrayprint import SubArrayFormat
import pandas as pd
import numpy as np

from pickle import dump, load
from math import isnan

from datetime import timedelta

from sklearn.preprocessing import MinMaxScaler


def load_data(fname="data/dataset_mood_smartphone.csv"):
    # Lees data in
    data = pd.read_csv(fname, index_col=0)

    # Verander time column naar dates
    data["time"] = pd.to_datetime(data["time"])
    data["hour"] = data["time"].dt.hour
    data["time"] = data["time"].dt.date

    return data


def get_featurelist():
    return dict(
        {
            "activity": "sum",
            "activity_night": "sum",
            "mood": "mean",
            "circumplex.arousal": "mean",
            "circumplex.valence": "mean",
            "appCat.builtin": "sum",
            "appCat.communication": "sum",
            "appCat.entertainment": "sum",
            "appCat.finance": "sum",
            "appCat.game": "sum",
            "appCat.office": "sum",
            "appCat.other": "sum",
            "appCat.social": "sum",
            "appCat.travel": "sum",
            "appCat.unknown": "sum",
            "appCat.utilities": "sum",
            "appCat.weather": "sum",
            "call": "sum",
            "screen": "count",
            "screen_night": "count",
            "sms": "sum",
        }
    )


def pivot_aggregate_data(data, excluded_features={}):
    """
    Averages all values per day and pivots
    pandas table to use day and user id as index.
    """

    # Default aggregation is taking the mean.
    new_df = data.pivot_table(
        values=["value"],
        columns="variable",
        aggfunc={"value": [sum, np.mean, "count"]},
        index=["id", "time"],
    )

    # print(new_df.columns)
    # Declare the aggregation type per variable
    featurelist = get_featurelist()
    to_keep = {k: v for k, v in featurelist.items() if k not in excluded_features}

    # Drop the other variables from the df
    selected_features = [x for x in new_df.columns if to_keep.get(x[2]) == x[1]]
    new_df = new_df[selected_features]

    # Drop the aggregation name multiindex level
    new_df = new_df.droplevel(1, axis=1)

    return new_df


def normalize_data(data, scaler_fp="scalers/scaler.pkl"):
    """
    Normalizes data using min-max scaling
    """

    # Set scaler and prepare copy of data
    scaler = MinMaxScaler()
    df = data.copy()

    # Seperate target variable
    target = ("value", "mood")
    cols_to_norm = df.columns.drop([target])

    # Scale variables seperately
    df[cols_to_norm] = scaler.fit_transform(df[cols_to_norm])
    df[[target]] = scaler.fit_transform(df[[target]])

    # If filepath specified- save mood scaler
    if isinstance(scaler_fp, str):
        dump(scaler, open(scaler_fp, "wb"))

    return df


def inverse_normalization(labels, scaler_fp="scalers/scaler.pkl"):
    """
    Loads scaler and applies inverserve
    normalisation to labels.
    """
    scaler = load(open(scaler_fp, "rb"))
    labels = scaler.inverse_transform([labels])

    return labels


def remove_moodless_days(data):
    data.dropna(subset=[("value", "mood")], inplace=True)


def impute_missing_values(data, ids):
    """
    Imputes missing values with the mean over
    that variable for that user.
    """
    # Missing values should be calculated for all categories but mood.
    rel_columns = data.columns.drop([("value", "mood")])

    # Loop over all users and all categories and replace NaNs with the mean
    # for that user and that category.
    for id in ids:
        for column in rel_columns:
            mean = data.loc[id][column].mean()
            mean = 0 if isnan(mean) else mean
            data.loc[id][column].fillna(value=mean, inplace=True)


def filter_outliers(raw_data, threshold=3600 * 3):
    """
    Filters outliers from raw data, by:
        - Removing negative app durations
        - Setting app durations above threshold to
          a specified threshold value

    Args:
        raw_data (df): The raw data.
        threshold (int): Threshold value (default 3h)

    Returns:
        df: The filtered data
    """
    df = raw_data.copy()

    # Get the appCat categories
    all_features = df.variable.unique()
    features = [x for x in all_features if "appCat" in x]

    # Remove negative values in the appCat categories
    non_neg_idx = df[(df.value < 0) & (df.variable.isin(features))].index
    df.drop(non_neg_idx, inplace=True)

    # Set values above threshold to threshold value
    # for all but the appCat.builtin category
    features.remove("appCat.builtin")
    outlier_idx = df[(df.value > threshold) & (df.variable.isin(features))].index
    df.loc[outlier_idx, "value"] = threshold

    return df


def add_night_features(filtered_data, start_time=2, end_time=5):
    """ 
    Duplicates activity & screen records 
    if between start_time and end_time.
    Adds _night to the variable name.
    """
    df = filtered_data.copy()

    # Extract night features
    night = df[(df.hour >= start_time) & (df.hour <= end_time) & (df.value > 0)]

    # Filter activity
    activity = night[night.variable == "activity"].copy()
    activity["variable"] = "activity_night"

    # Filter screentime
    screen = night[night.variable == "screen"].copy()
    screen["variable"] = "screen_night"

    # Add back to DF
    merged = pd.concat([df, activity, screen])

    # Check if everything went well..
    assert df.shape[1] == merged.shape[1]
    assert merged.shape[0] == df.shape[0] + activity.shape[0] + screen.shape[0]

    return merged


def preprocess_raw_data(normalize=True, sleep_indication=True, excluded_features={}):
    """
    Takes raw data as input, imputes missing values, normalizes the data
    and returns the updated dataset.
    """

    raw_data = load_data()
    filtered_data = filter_outliers(raw_data)
    ids = list(set(filtered_data["id"]))
    ids.sort()

    if sleep_indication:
        filtered_data = add_night_features(filtered_data)

    data = pivot_aggregate_data(filtered_data, excluded_features=excluded_features)
    remove_moodless_days(data)
    impute_missing_values(data, ids)

    if normalize:
        data = normalize_data(data)

    return data


def get_consecutive_days(data, no_days=5, overlap=True):
    # Create timdelta depending on the number of days
    td = timedelta(no_days)
    td_1 = timedelta(1)

    # Initialize list for start-and end day tuples.
    start_end_list = []

    # Loop over all rows in the dataframe
    index = 0
    while True:
        patient, date = data.index[index]
        new_index = index + no_days

        # Check for index out of bounds
        if new_index >= len(data):
            break

        # Get the patient and date if we go down no_days rows
        new_patient, new_date = data.index[new_index]

        # Check if going down a number of rows result in the
        # same id and same difference in date, then we want
        # this datapoint and we save it.
        if (patient, date + td) == (new_patient, new_date):
            start_end_target = (
                (patient, date),
                (patient, new_date - td_1),
                (patient, new_date),
            )
            start_end_list.append(start_end_target)

        # If there is overlap we go one row down, otherwise
        # we make sure to jump to the last day of the current
        # sequence
        index += 1 if overlap else no_days - 1

    return start_end_list


def get_aggregated_data(no_days=5, excluded_features={}):

    data = preprocess_raw_data(excluded_features=excluded_features)

    start_end_list = get_consecutive_days(data, no_days=no_days)
    instances, labels = [], []
    for start, end, target in start_end_list:
        data_slice = data.loc[start:end]
        instance = np.array(data_slice.mean())
        label = data.loc[target, ("value", "mood")]
        instances.append(instance), labels.append(label)

    return instances, labels


def get_baseline_data(no_days=5):
    data = preprocess_raw_data(normalize=False)

    # We still use the no_days variable here because
    # we want to use the same test days as for the
    # other models
    start_end_list = get_consecutive_days(data, no_days=no_days)
    instances, labels = [], []
    for _, last_day, target in start_end_list:
        instance = data.loc[last_day, ("value", "mood")]
        label = data.loc[target, ("value", "mood")]
        instances.append(instance), labels.append(label)

    return instances, labels


# Read & Aggregate data
# if script is called directly
if __name__ == "__main__":
    instances, labels = get_aggregated_data()

# %%
