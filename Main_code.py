# Code for Data Mining Techniques project period 5 2020/2021 VU

# Import packages
import pandas as pd
from math import isnan


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
        values=["value"], columns="variable", index=["time", "id"]
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


def impute_missing_values(data, ids):
    """
    Imputes missing values with the mean over
    that variable for that user.
    """
    mean_df = pd.DataFrame(index=ids, columns=data.value.columns)
    for id in ids:
        for column in data:
            mean = data.xs(id, level="id")[column].mean()
            data.xs(id, level="id")[column].fillna(value=mean, inplace=True)


raw_data = load_data()
ids = list(set(raw_data["id"]))
ids.sort()
data = pivot_average_data(raw_data)
norm_data = normalize_data(data)
