# Data Mining Techniques 2020/2021 VU

# %% Import packages
# Additionally requires tabulate for df.to_markdown()
import pandas as pd
import numpy as np
from math import ceil
import matplotlib.pyplot as plt

# %% Helper functions


def load_process_data(path):
    raw_data = pd.read_csv(path, index_col=0)
    raw_data["datetime"] = pd.to_datetime(raw_data["time"])
    raw_data["date"] = raw_data["datetime"].dt.date
    raw_data["hour"] = raw_data["datetime"].dt.hour
    return raw_data


def describe_userstats(df):
    unique_users = df.id.unique()
    print(f"{len(unique_users)} unique users:\n")
    print(f'{", ".join(unique_users)}\n')

    stat_df = df.groupby(["id"]).aggregate({"variable": ["count"]})
    instances_per_user = stat_df[("variable", "count")]
    print(
        f"Mean instances per user: {round(np.mean(instances_per_user))}±{round(np.std(instances_per_user))}"
    )


def describe_variables(df, merge_appcats=False, output_format="markdown"):
    base_stats = []

    if merge_appcats:
        df["variable"] = df["variable"].apply(
            lambda x: "appCat.n" if "appCat" in x else x
        )
        df["variable"] = df["variable"].apply(
            lambda x: "circumplex.A/V" if "circumplex" in x else x
        )

    for var in df.variable.unique():
        subsetseries = raw_data[raw_data["variable"] == var].value
        description_dict = subsetseries.describe().round(2).to_dict()
        description_dict["variable"] = var
        base_stats.append(description_dict)

    base_stats_df = pd.DataFrame(base_stats)
    base_stats_df = base_stats_df.set_index("variable")

    if output_format == "markdown":
        return base_stats_df.to_markdown()
    if output_format == "latex":
        return base_stats_df.to_latex()

    return base_stats_df


def plot_mood_per_hour(df, output_path, size=(9, 3)):
    """
    Plot mood per hour of day
    """
    df_mood_only = df[df["variable"] == "mood"]
    plt.rcParams["figure.figsize"] = size

    # %% Mood recordings per hour of day
    df_mood_hourly = df_mood_only[["hour", "id", "value"]].groupby(["hour"])
    df_mood_hourly = df_mood_hourly.aggregate({"id": "count", "value": "mean"})
    df_mood_hourly.reset_index(inplace=True)

    plt.xlabel("Hour of day")
    plt.ylabel("# Mood recordings")
    plt.title("Number of mood recordings per hour of day.")
    plt.bar(df_mood_hourly.hour, df_mood_hourly.id)

    plt.savefig(output_path, dpi=300, transparent=False)


def plot_mood_per_date(df, output_path, size=(9, 3)):
    """
    Plot mood per date
    """
    df_mood_only = df[df["variable"] == "mood"]
    plt.rcParams["figure.figsize"] = size

    df_mood_daily = df_mood_only[["date", "id", "value"]].groupby(["date"])
    df_mood_daily = df_mood_daily.aggregate({"id": "count", "value": "mean"})
    df_mood_daily.reset_index(inplace=True)

    plt.xlabel("Date")
    plt.xticks(rotation=33)
    plt.ylabel("# Mood recordings")
    plt.title("Number of mood recordings per date.")
    plt.bar(df_mood_daily.date, df_mood_daily.id)

    plt.savefig(output_path, dpi=300, transparent=False)


def plot_sensors_per_hour(df, output_path, size=(9, 3)):
    """
    """
    df_sensors_only = df[df["variable"] != "mood"]

    # %% Sensor recordings per hour of day
    df_sensors_hourly = df_sensors_only[["hour", "id", "value"]].groupby(["hour"])
    df_sensors_hourly = df_sensors_hourly.aggregate({"id": "count", "value": "mean"})
    df_sensors_hourly.reset_index(inplace=True)

    plt.xlabel("Hour of day")
    plt.ylabel("# Sensor recordings")
    plt.title("Number of sensor recordings per hour of day.")
    plt.rcParams["figure.figsize"] = size
    plt.bar(df_sensors_hourly.hour, df_sensors_hourly.id)

    plt.savefig(output_path, dpi=300, transparent=False)


def plot_sensors_per_date(df, output_path, size=(9, 3)):
    """
    """

    df_sensors_only = df[df["variable"] != "mood"]
    df_sensors_daily = df_sensors_only[["date", "id", "value"]].groupby(["date"])
    df_sensors_daily = df_sensors_daily.aggregate({"id": "count", "value": "mean"})
    df_sensors_daily.reset_index(inplace=True)

    plt.xlabel("Date")
    plt.xticks(rotation=33)
    plt.ylabel("# Sensor recordings")
    plt.title("Number of sensor recordings per date.")
    plt.rcParams["figure.figsize"] = size
    plt.bar(df_sensors_daily.date, df_sensors_daily.id)

    plt.savefig(output_path, dpi=300, transparent=False)


def plot_variable_histograms(df, output_path, size=(40, 40)):
    variable_count = len(df.variable.unique())
    cols, rows = 4, ceil(variable_count / 4)
    fig, axs = plt.subplots(rows, cols, figsize=size)
    axs = axs.ravel()
    # print(fig, axs)

    for idx, var in enumerate(df.variable.unique()):
        sub_df = df[(df.variable == var) & (df.value < 3600 * 4) & (df.value > -2.1)]
        axs[idx].hist(sub_df.value, bins=100)
        axs[idx].set_title(f"{var}")
        # axs[idx].set_xscale("log")
        # print(sub_df)
        # print(var)
        # break

    fig.savefig(output_path, dpi=300, transparent=False)


# %% Load data
if __name__ == "__main__":

    raw_data = load_process_data("data/dataset_mood_smartphone.csv")
    # %% Extract stats per user
    describe_userstats(raw_data)
    # %%
    print(describe_variables(raw_data, merge_appcats=True, output_format="latex"))
    # %%
    plot_mood_per_hour(raw_data, "plots/mood_per_hour.png")
    # %%
    plot_mood_per_date(raw_data, "plots/mood_per_date.png")
    # %%
    plot_sensors_per_hour(raw_data, "plots/sensor_per_hour.png")
    # %%
    plot_sensors_per_date(raw_data, "plots/sensor_per_date.png")
    # %%
    plot_variable_histograms(raw_data, "plots/variable_histograms.png")
    # %%

    # %%
