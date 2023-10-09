import random
from typing import List

import numpy as np
import pandas as pd

from SurvSet.data import SurvLoader
from pycox.datasets import metabric

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_selector
from sklearndf.pipeline import PipelineDF
from sklearndf.transformation import (
    OneHotEncoderDF,
    ColumnTransformerDF,
    SimpleImputerDF,
    StandardScalerDF,
)


def get_metabric_df():
    mdf = metabric.read_df()
    mdf = mdf.rename(
        columns={
            "duration": "time",
            "x0": "num_x0",
            "x1": "num_x1",
            "x2": "num_x2",
            "x3": "num_x3",
            "x4": "fac_x4",
            "x5": "fac_x5",
            "x6": "fac_x6",
            "x7": "fac_x7",
            "x8": "num_x8",
        }
    )
    return mdf


def get_flamby_df():
    raise ValueError("The TCGA-BRCA dataset is available here: https://github.com/owkin/FLamby")


DATASETS_BEYOND_SURVSET = {
    "metabric": get_metabric_df,
    "flamby": get_flamby_df,
}


def get_dataset_names() -> List[str]:
    surv_loader = SurvLoader()
    names = surv_loader.df_ds[~surv_loader.df_ds["is_td"]].drop(columns=["is_td"])
    names = names["ds"].to_list()
    for d in DATASETS_BEYOND_SURVSET:
        names.append(d)
    return names


def get_dataframe(name: str) -> pd.DataFrame:
    if name in DATASETS_BEYOND_SURVSET:
        return DATASETS_BEYOND_SURVSET[name]()
    return SurvLoader().load_dataset(name)["df"]


def _get_preprocess_transformer() -> ColumnTransformerDF:
    sel_fac = make_column_selector(pattern="^fac\\_")
    enc_fac = PipelineDF(
        steps=[
            (
                "ohe",
                OneHotEncoderDF(
                    sparse=False, drop="if_binary", handle_unknown="ignore"
                ),
            )
        ]
    )
    sel_num = make_column_selector(pattern="^num\\_")
    enc_num = PipelineDF(
        steps=[
            ("impute", SimpleImputerDF(strategy="median")),
            ("scale", StandardScalerDF()),
        ]
    )
    tr = ColumnTransformerDF(
        transformers=[("ohe", enc_fac, sel_fac), ("s", enc_num, sel_num)]
    )
    return tr


def _split_dataframe(df: pd.DataFrame, split_size=0.2) -> (pd.DataFrame, pd.DataFrame):
    return train_test_split(df, stratify=df["event"], test_size=split_size)


def preprocess_dataframe(
    df: pd.DataFrame, split_size=0.2
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    tr = _get_preprocess_transformer()
    df_train, df_test = _split_dataframe(df, split_size=split_size)
    X_train = tr.fit_transform(df_train).to_numpy().astype(np.float32)
    X_test = tr.transform(df_test).to_numpy().astype(np.float32)
    sksurv_type = [("event", bool), ("time", float)]
    y_train = np.array(
        [(e, t) for e, t in zip(df_train["event"], df_train["time"])], dtype=sksurv_type
    )
    y_test = np.array(
        [(e, t) for e, t in zip(df_test["event"], df_test["time"])], dtype=sksurv_type
    )
    return X_train, X_test, y_train, y_test


def get_split_data(dataset: str, val_split: float, test_split: float, random_state: int):
    df = get_dataframe(dataset)
    X_train, X_test, y_train, y_test = preprocess_dataframe(df, split_size=test_split)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_split, random_state=random_state
    )

    X_test = X_test[
        (y_test["time"] >= min(y_train["time"]))
        & (y_test["time"] <= max(y_train["time"]))
    ]
    y_test = y_test[
        (y_test["time"] >= min(y_train["time"]))
        & (y_test["time"] <= max(y_train["time"]))
    ]

    return X_train, y_train, X_val, y_val, X_test, y_test
