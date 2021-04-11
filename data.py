# Code for Data Mining Techniques project period 5 2020/2021 VU

# Import packages
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


def preprocess_raw_data(data):
    """
    Takes raw data as input, imputes missing values, normalizes the data
    and returns the updated dataset.
    """
    raw_data = load_data()
    ids = list(set(raw_data["id"]))
    ids.sort()
    data = pivot_average_data(raw_data)
    remove_moodless_days(data)
    impute_missing_values(data, ids)
    norm_data = normalize_data(data)

    return norm_data


def get_consecutive_days(data, ids, no_days=5):
    # Create timdelta depending on the number of days
    td = timedelta(no_days)

    # Initialize list for start-and end day tuples.
    start_end_list = []

    for index, ((id, date), content) in enumerate(data.iterrows()):
        # Check if going down a number of rows result in the
        # same difference in date.
        new_index = index + no_days
        if new_index >= len(data):
            break
        new_date = date + td
        if data.index[new_index] == (id, new_date):
            start_end = ((id, date), (id, new_date))
            start_end_list.append(start_end)

    return start_end_list


def get_aggregated_split(data):
    return 0
