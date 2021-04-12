# Code for Data Mining Techniques project period 5 2020/2021 VU

# %% Import packages
import pandas as pd
from math import isnan
import numpy as np
from datetime import date, timedelta


def load_data(fname="data/dataset_mood_smartphone.csv"):
    # Lees data in
    data = pd.read_csv(fname, index_col=0)

    # Verander time column naar dates
    data["time"] = pd.to_datetime(data["time"])
    data["time"] = data["time"].dt.date

    return data


def pivot_average_data(data):
    """
    Averages all values per day and pivots
    pandas table to use day and user id as index.
    """
    # Default aggregation is taking the mean.
    new_dataset = data.pivot_table(
        values=["value"], columns="variable", index=["id", "time"]
    )

    return new_dataset


def normalize_data(data):
    """
    Normalizes data using min-max scaling
    """
    new_dataset = data.copy()
    # All values except the 'call' and 'sms' column
    # need to be normalised.
    rel_columns = data.columns.drop([("value", "sms"), ("value", "call")])
    for column in rel_columns:
        new_dataset[column] = (data[column] - data[column].min()) / (
            data[column].max() - data[column].min()
        )

    return new_dataset


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
    outlier_idx = df[
        (df.value > threshold) & (df.variable.isin(features))
    ].index
    df.loc[outlier_idx, "value"] = threshold

    return df


def preprocess_raw_data(normalize=True):
    """
    Takes raw data as input, imputes missing values, normalizes the data
    and returns the updated dataset.
    """
    raw_data = load_data()
    filtered_data = filter_outliers(raw_data)
    ids = list(set(filtered_data["id"]))
    ids.sort()
    data = pivot_average_data(filtered_data)
    remove_moodless_days(data)
    impute_missing_values(data, ids)
    if normalize:
        data = normalize_data(data)

    return data


# def get_consecutive_days(data, ids, no_days=5):
#     # Create timdelta depending on the number of days
#     td = timedelta(no_days)

#     # Initialize list for start-and end day tuples.
#     start_end_list = []

#     for index, ((patient, date), content) in enumerate(data.iterrows()):
#         # Check if going down a number of rows result in the
#         # same difference in date.
#         new_index = index + no_days
#         if new_index >= len(data):
#             break
#         new_date = date + td
#         if data.index[new_index] == (patient, new_date):
#             start_end = ((patient, date), (patient, new_date))
#             start_end_list.append(start_end)

#     return start_end_list


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


def get_aggregated_data(no_days=5):
    data = preprocess_raw_data()
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
        instances.append(instance), label.append(label)

    return instances, labels


# Read & Aggregate data
# if script is called directly
if __name__ == "__main__":
    instances, labels = get_aggregated_data()
# %%
