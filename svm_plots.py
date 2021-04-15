# %%

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_pickle("results/SVM_gridsearch_and_features.pkl")
selection = df[
    (df.features == "mood, appCat.finance")
    | (df.features == "mood, appCat.finance, appCat.entertainment")
    | (df.features == "mood, appCat.finance, appCat.entertainment, call")
    | (df.features == "mood, appCat.finance, appCat.entertainment, call, sms")
    | (
        df.features
        == "mood, appCat.finance, appCat.entertainment, call, sms, appCat.social"
    )
    | (
        df.features
        == "mood, appCat.finance, appCat.entertainment, call, sms, appCat.social, appCat.game"
    )
]

# %%
fig, ax = plt.subplots(figsize=(8, 3))

plot = sns.lineplot(ax=ax, data=selection, x="feature_count", y="mse")

plot.set_xticks([x for x in range(2, 8)])
plot.set_xticklabels(
    [
        "2\nmood\nfinance",
        "3\nmood\nfinance\nentertainment",
        "4\nmood\nfinance\nentertainment\ncall",
        "5\nmood\nfinance\nentertainment\ncall\nsms",
        "6\nmood\nfinance\nentertainment\ncall\nsms\nsocial",
        "7\nmood\nfinance\nentertainment\ncall\nsms\nsocial\ngame",
    ]
)
plot.set_xlabel("Number of features")
plot.set_ylabel("Score (MSE)")
plot.set_title("Score progression bottom-up feature selection.")
plot.set_ylim(0.325, 0.45)


# mood, appCat.finance, appCat.entertainment, call, sms, appCat.social, appCat.game
# df["best_params"] = df["best_params"].apply(lambda x: str(x))
# df = df[df.aggregated_days <= 14]

# stats = df.groupby(["features"], as_index=False).agg(
#     {"mse": ["mean", "var", "std"]}  # , "std"]}
# )

# stats.columns = stats.columns.map("|".join).str.strip("|")

# # result = df.groupby(['a'], as_index=False).agg(
# #                       {'c':['mean','std'],'b':'first', 'd':'first'})


# %% Full plot
df30 = pd.read_pickle("results/MSE:0.3935_RANGE_1-20_FEATURES_ALL_30x.pkl")
# df30 = df30.groupby(["aggregated_days"]).agg({"mse": ["mean", "std"]})
# df30.columns = df30.columns.map("|".join).str.strip("|")


# ax1 = df30.plot.scatter(x='aggregated_days',
#                       y='mse',
#                       c='DarkBlue')

fig, ax = plt.subplots(figsize=(8, 3))

plot = sns.lineplot(ax=ax, data=df30, x="aggregated_days", y="mse")

plot.set_xticks([x for x in range(1, 21)])
plot.set_xticklabels([x for x in range(1, 21)])
plot.set_xlabel("Aggregation level (days)")
plot.set_ylabel("Score (MSE)")
plot.set_title("SVR score (MSE) per aggregation level (days) - 100 runs.")
plot.set_ylim(0.325, 0.45)

# %%


# %%
