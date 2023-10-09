from src.models.coxph import train_cox
from src.models.deephit import train_deephit
from src.models.deepsurv import train_ds
from src.models.logistic_hazard import train_nnet
from src.models.nmtlr import train_nmtlr

__MODELS__ = [
    "coxph",
    "deepsurv",
    "deephit",
    "logistic_hazard",
    "nmtlr",
]


def train_model(
        model: str,
        X_train,
        y_train,
        X_val,
        y_val,
        hidden_features,
        splits,
        batch_size,
        epochs,
        patience,
):
    if model not in __MODELS__:
        raise ValueError(f"Unknown model {model}.")
    if model == "coxph":
        return train_cox(
            X_train,
            y_train,
            X_val,
            y_val,
            hidden_features,
            splits,
            batch_size,
            epochs,
            patience,
        )
    if model == "deepsurv":
        return train_ds(
            X_train,
            y_train,
            X_val,
            y_val,
            hidden_features,
            splits,
            batch_size,
            epochs,
            patience,
        )
    if model == "deephit":
        return train_deephit(
            X_train,
            y_train,
            X_val,
            y_val,
            hidden_features,
            splits,
            batch_size,
            epochs,
            patience,
        )
    if model == "logistic_hazard":
        return train_nnet(
            X_train,
            y_train,
            X_val,
            y_val,
            hidden_features,
            splits,
            batch_size,
            epochs,
            patience,
        )
    if model == "nmtlr":
        return train_nmtlr(
            X_train,
            y_train,
            X_val,
            y_val,
            hidden_features,
            splits,
            batch_size,
            epochs,
            patience,
        )
