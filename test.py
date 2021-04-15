# %%
import pandas as pd
from pickle import load

# res = pd.read_pickle("results/SVM.pkl")

# for x in res.best_params:
#     print(x)
path = "models/svm_all_0.3369.pkl"
model = load(open(path, "rb"))

# %%
