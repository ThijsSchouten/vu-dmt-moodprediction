# Code for Data Mining Techniques project period 5 2020/2021 VU

# Import packages
import pandas as pd

# Lees data in
data = pd.read_csv("data/dataset_mood_smartphone.csv", index_col = 0)
data = data.pivot_table(values = ["value"], columns = "variable", index = ["time","id"])
print(data.index.shape)
#data.pivot(index = "time")
#print(data)