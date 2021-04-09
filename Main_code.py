# Code for Data Mining Techniques project period 5 2020/2021 VU

# Import packages
import pandas as pd

def load_data(fname="data/dataset_mood_smartphone.csv")
    # Lees data in
    data = pd.read_csv("data/dataset_mood_smartphone.csv", index_col=0)

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
    new_dataset = data = data.pivot_table(
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
