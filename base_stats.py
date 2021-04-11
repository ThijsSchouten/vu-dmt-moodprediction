# Data Mining Techniques 2020/2021 VU

# %% Import packages
# Additionally requires tabulate for df.to_markdown()
import pandas as pd
import matplotlib.pyplot as plt

# %%
def load_process_data(path):
    raw_data = pd.read_csv(path, index_col=0)
    raw_data["datetime"] = pd.to_datetime(raw_data["time"])
    raw_data["date"] = raw_data["datetime"].dt.date
    raw_data["hour"] = raw_data["datetime"].dt.hour
    return raw_data

# %% Count&list unique users
def describe_userstats(df):
    unique_users = df.id.unique()
    print(f'{len(unique_users)} unique users:\n')
    print(f'{", ".join(unique_users)}\n')


# %% 
def describe_variables(df, markdown=True):
    base_stats = []
    for var in df.variable.unique():
        subsetseries = raw_data[raw_data['variable'] == var].value
        description_dict = subsetseries.describe().round(2).to_dict()
        description_dict['variable'] = var
        base_stats.append(description_dict)


    base_stats_df = pd.DataFrame(base_stats)
    base_stats_df = base_stats_df.set_index('variable')
    if markdown:
        return base_stats_df.to_markdown()

    return base_stats_df

# %% 
def plot_mood_per_hour(df, output_path, size=(9,3)):
    '''
    Plot mood per hour of day
    '''
    df_mood_only = df[df['variable'] == 'mood']
    plt.rcParams["figure.figsize"] = size    

    # %% Mood recordings per hour of day
    df_mood_hourly = df_mood_only[['hour', 'id', 'value']].groupby(['hour'])
    df_mood_hourly = df_mood_hourly.aggregate({'id': 'count', 'value': 'mean'})
    df_mood_hourly.reset_index(inplace=True)

    plt.xlabel('Hour of day')
    plt.ylabel('# Mood recordings')
    plt.title('Number of mood recordings per hour of day.')
    plt.bar(df_mood_hourly.hour, df_mood_hourly.id)

    plt.savefig(output_path, dpi=300, transparent=False)



# %% Mood recordings per day
def plot_mood_per_date(df, output_path, size=(9,3)):
    '''
    Plot mood per date
    '''
    df_mood_only = df[df['variable'] == 'mood']
    plt.rcParams["figure.figsize"] = size

    df_mood_daily = df_mood_only[['date', 'id', 'value']].groupby(['date'])
    df_mood_daily = df_mood_daily.aggregate({'id': 'count', 'value': 'mean'})
    df_mood_daily.reset_index(inplace=True)

    plt.xlabel('Date')
    plt.xticks(rotation=33)
    plt.ylabel('# Mood recordings')
    plt.title('Number of mood recordings per date.')
    plt.bar(df_mood_daily.date, df_mood_daily.id)

    plt.savefig(output_path, dpi=300, transparent=False)


# %% Sensor info per {hour, day} plots
def plot_sensors_per_hour(df, output_path, size=(9,3)):
    '''
    '''
    df_sensors_only = df[df['variable'] != 'mood']

    # %% Sensor recordings per hour of day
    df_sensors_hourly = df_sensors_only[['hour', 'id', 'value']].groupby(['hour'])
    df_sensors_hourly = df_sensors_hourly.aggregate({'id': 'count', 'value': 'mean'})
    df_sensors_hourly.reset_index(inplace=True)

    plt.xlabel('Hour of day')
    plt.ylabel('# Sensor recordings')
    plt.title('Number of sensor recordings per hour of day.')
    plt.rcParams["figure.figsize"] = (9,3)
    plt.bar(df_sensors_hourly.hour, df_sensors_hourly.id)

    plt.savefig(output_path, dpi=300, transparent=False)



# %% Sensor recordings per day
def plot_sensors_per_date(df, output_path, size=(9,3)):

    df_sensors_only = df[df['variable'] != 'mood']
    df_sensors_daily = df_sensors_only[['date', 'id', 'value']].groupby(['date'])
    df_sensors_daily = df_sensors_daily.aggregate({'id': 'count', 'value': 'mean'})
    df_sensors_daily.reset_index(inplace=True)

    plt.xlabel('Date')
    plt.xticks(rotation=33)
    plt.ylabel('# Sensor recordings')
    plt.title('Number of sensor recordings per date.')
    plt.rcParams["figure.figsize"] = (9,3)
    plt.bar(df_sensors_daily.date, df_sensors_daily.id)

    plt.savefig(output_path, dpi=300, transparent=False)


# %% Load data 
raw_data = load_process_data("data/dataset_mood_smartphone.csv")
# %% Extract stats per user
describe_userstats(raw_data)
# %% 
describe_variables(raw_data)
# %% 
plot_mood_per_hour(raw_data, 'plots/mood_per_hour.png')
# %%
plot_mood_per_date(raw_data, 'plots/mood_per_date.png')
# %%
plot_sensors_per_hour(raw_data, 'plots/sensor_per_hour.png')
# %%
plot_sensors_per_date(raw_data, 'plots/sensor_per_date.png')