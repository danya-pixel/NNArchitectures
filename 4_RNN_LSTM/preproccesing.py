import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy.lib.stride_tricks import sliding_window_view


def apply_enconding(df):
    days = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }

    df["Day_of_week"] = df["Day_of_week"].map(days)
    df["WeekStatus"] = df["WeekStatus"].map({"Weekday": 0, "Weekend": 1})
    df["Load_Type"] = df["Load_Type"].map(
        {"Light_Load": 0, "Medium_Load": 1, "Maximum_Load": 2}
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    day_len_seconds = 24 * 60 * 60
    year_len_seconds = (365.2425) * day_len_seconds

    df["Time_sin"] = np.sin(
        (df["date"] - df["date"].min()).dt.total_seconds()
        * (2 * np.pi / day_len_seconds)
    )
    df["Date_sin"] = np.sin(
        (df["date"] - df["date"].min()).dt.total_seconds()
        * (2 * np.pi / year_len_seconds)
    )
    return df


def get_feature_cols(df):
    df.drop(columns=["date"], inplace=True)
    feature_columns = list(df.columns)
    feature_columns.remove("Usage_kWh")
    df[feature_columns] = MinMaxScaler().fit_transform(df[feature_columns])
    return df


def train_val_test_seq(df):
    train_val_len = int(len(df) * 0.8)
    test_len = len(df) - train_val_len
    train_len = int(train_val_len * 0.8)
    val_len = train_val_len - train_len

    train_df = df[:train_len]
    val_df = df[train_len : train_len + val_len]
    test_df = df[train_len + val_len : train_len + val_len + test_len]

    return train_df, val_df, test_df


def apply_sliding_window(df, seq_len, target_size, target_col):
    features = df.to_numpy()

    targets = df[target_col].to_numpy()

    x = sliding_window_view(features[0:-target_size], window_shape=(seq_len, 12))
    y = targets[seq_len:]
    x = np.squeeze(x, axis=1)
    return x.copy(), y.reshape(-1, 1).copy()
