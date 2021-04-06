# Code for Data Mining Techniques project period 5 2020/2021 VU

# Import packages
import pandas as pd

# Lees data in
data = pd.read_csv("data/dataset_mood_smartphone.csv", index_col = 0)
# Verander time column naar dates
data["time"] = pd.to_datetime(data["time"])
data["time"] = data["time"].dt.date
# Pivot naar elke rij 1 datum met alle variabelen
data = data.pivot_table(values = ["value"], columns = "variable", index = ["time","id"])
# Aggrereer naar dag niveau door variabelen te averagen per dag
# Deze stap werkt nog niet, hieronder is een probeersel
#data = data.groupby([data.index.levels[0].values, data.index.levels[1].values], as_index = False).mean()
print(data.index.levels[0].values)