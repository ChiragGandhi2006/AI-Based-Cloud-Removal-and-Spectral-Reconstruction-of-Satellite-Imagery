import numpy as np
import os
import sklearn.model_selection as ms


def geographic_split(df, test_ratio=0.15, val_ratio=0.15, random_state=42):
    groups = df["scene"].unique()
    train_val_groups, test_groups = ms.train_test_split(
        groups, test_size=test_ratio, random_state=random_state
    )
    train_groups, val_groups = ms.train_test_split(
        train_val_groups, test_size=val_ratio / (1 - test_ratio), random_state=random_state
    )

    train_df = df[df["scene"].isin(train_groups)]
    val_df = df[df["scene"].isin(val_groups)]
    test_df = df[df["scene"].isin(test_groups)]

    return train_df, val_df, test_df


def random_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    indices = np.random.RandomState(random_state).permutation(len(df))
    train_sz = int(len(df) * train_ratio)
    val_sz = int(len(df) * val_ratio)

    train_idx = indices[:train_sz]
    val_idx = indices[train_sz : train_sz + val_sz]
    test_idx = indices[train_sz + val_sz :]

    return df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]