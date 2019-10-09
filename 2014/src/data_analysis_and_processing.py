from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# ## Data Loading


consumption = pd.read_csv("../data/Train - Part Consumption.csv")


usage = pd.read_csv("../data/Train - Usage.csv")


failures = pd.read_csv("../data/Train - Failures.csv")


c = Counter(consumption.Reason)
reason_count = c.most_common()
print(reason_count)


c = Counter(failures.Asset)
failures_count = c.most_common()
print(failures_count)


# ## Data Preprocessing


consumption.Time /= 730.0
usage.Time /= 730.0
failures.Time /= 730.0


indexes = consumption.query("Quantity <= 0").index
consumption.drop(indexes, inplace=True)
consumption.reset_index(inplace=True)


# ## Training Data construction


train_df = consumption.copy()
train_df.drop("index", axis=1, inplace=True)


print(train_df.shape)


train_df["Failure"] = [False] * train_df.shape[0]
train_df["Time_failure"] = [0] * train_df.shape[0]
train_df["Time_diff"] = [0] * train_df.shape[0]
train_df["Usage_on_failure"] = [0] * train_df.shape[0]


for fail_asset, fail_time in tqdm(list(failures.itertuples(index=False, name=None))):
    possible_cons = consumption.query("Time <= @fail_time and Asset == @fail_asset")

    usage_next = usage.query("Time >= @fail_time and Asset == @fail_asset").head(1)
    usage_prev = usage.query("Time <= @fail_time and Asset == @fail_asset").tail(1)

    usage_failure_value = 0
    if len(usage_next) > 0 and len(usage_prev) > 0:
        usage_prev_time = usage_prev.iloc[0, 1]
        usage_prev_value = usage_prev.iloc[0, 2]
        usage_next_time = usage_next.iloc[0, 1]
        usage_next_value = usage_next.iloc[0, 2]

        if usage_next_time - usage_prev_time > 0:
            usage_failure_value = fail_time - usage_prev_time
            usage_failure_value /= usage_next_time - usage_prev_time
            usage_failure_value *= usage_next_value - usage_prev_value
            usage_failure_value += usage_prev_value
        else:
            usage_failure_value = usage_next_value

    inserted_parts = []
    for index, cons_part in possible_cons[::-1].iterrows():
        if cons_part.Part not in inserted_parts:
            train_df.loc[index, "Failure"] = True
            train_df.loc[index, "Time_failure"] = fail_time
            train_df.loc[index, "Time_diff"] = fail_time - consumption.loc[index, "Time"]
            train_df.loc[index, "Usage_on_failure"] = usage_failure_value
            inserted_parts.append(cons_part.Part)

# train_df = pd.read_csv("../data/train_features.csv")


print(train_df.query("Failure==True"))


train_df.to_csv("../data/train_features.csv", index=False)


# This train set above does not really express others examples by considering some negative failures instances. So, I'll do this on Usage_on_failure column, by just measuring the usage on the Time column.


train_df.drop(["Time_failure", "Time_diff", "Usage_on_failure"], axis=1, inplace=True)


for index, row in tqdm(train_df[["Asset", "Time"]].iterrows()):
    time = row.Time
    asset = row.Asset
    usage_next = usage.query("Time >= @time and Asset == @asset").head(1)
    usage_prev = usage.query("Time <= @time and Asset == @asset").tail(1)

    usage_value = 0
    if len(usage_next) > 0 and len(usage_prev) > 0:
        usage_prev_time = usage_prev.iloc[0, 1]
        usage_prev_value = usage_prev.iloc[0, 2]
        usage_next_time = usage_next.iloc[0, 1]
        usage_next_value = usage_next.iloc[0, 2]

        if usage_next_time - usage_prev_time > 0:
            usage_value = row.Time - usage_prev_time
            usage_value /= usage_next_time - usage_prev_time
            usage_value *= usage_next_value - usage_prev_value
            usage_value += usage_prev_value
        else:
            usage_value = usage_next_value

    train_df.loc[index, "Usage_on_time"] = usage_value


train_df.to_csv("../data/train_features_usage.csv", index=False)
