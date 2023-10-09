from collections import OrderedDict
from typing import List

import numpy as np
import torch
from sksurv.metrics import (
    integrated_brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
)

from src.anchor_surv_fn import process_times_anchors, interpolate_points, _compute_tangents


def to_pycox_labels(y: np.array) -> (np.array, np.array):
    return y["time"].copy(), y["event"].copy()


def to_sksurv_labels(y_time: np.array, y_event: np.array) -> np.array:
    sksurv_type = [("event", bool), ("time", float)]
    return np.array([(e, t) for e, t in zip(y_event, y_time)], dtype=sksurv_type)


def get_parameters(model: torch.nn.Module) -> List[np.array]:
    return [val.cpu().numpy() for _, val in model.net.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: List[np.array]) -> None:
    params_dict = zip(model.net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.net.load_state_dict(state_dict, strict=True)


def compute_preds(anchors, values, times, mode):
    num_survs = len(values)
    num_times = len(times)
    preds = np.empty((num_survs, num_times), dtype=np.float32)

    for i in range(num_survs):
        a, v = process_times_anchors(anchors, values[i])
        if mode == "spline":
            tangents = _compute_tangents(a, v)
        for j in range(num_times):
            assert np.isscalar(times[j]), times[j]
            assert len(a.shape) == 1, a.shape
            if mode == "spline":
                preds[i, j] = interpolate_points(times[j], a, v, mode, tangents=tangents)
            else:
                preds[i, j] = interpolate_points(times[j], a, v, mode)

    preds = np.nan_to_num(preds, nan=0.5)
    preds = np.clip(preds, 0.0, 1.0)
    preds = preds * (1.0 - 1e-8) + 1e-8
    return preds


def evaluate(
        model: torch.nn.Module,
        y_train: np.array,
        X_test: np.array,
        y_test: np.array,
        mode: str,
) -> (float, float, float, float):
    # convert to pycox labels
    _y_test = to_pycox_labels(y_test)

    # predict survival functions
    surv_df_test = model.predict_surv_df(X_test)
    anchors = surv_df_test.index.to_numpy()
    values = [surv_df_test[i].to_numpy() for i in range(len(X_test))]

    # evaluate times of interest
    sorted_times = np.sort(np.unique(y_test["time"]))
    min_time, max_time = sorted_times[1], sorted_times[-2]
    times = np.linspace(max(min_time, np.percentile(y_test["time"], 25)),
                        min(max_time, np.percentile(y_test["time"], 75)), 1000)

    # evaluate predictions
    preds = compute_preds(anchors, values, times, mode)
    risks = -np.log(preds)
    cum_risks = np.sum(risks, axis=1)

    # evaluate metrics
    c_index = concordance_index_censored(y_test["event"], y_test["time"], cum_risks)[0]
    c_index_ipcw = concordance_index_ipcw(y_train, y_test, cum_risks)[0]
    ibs = integrated_brier_score(y_train, y_test, preds, times)
    auc = np.nanmean(cumulative_dynamic_auc(y_train, y_test, risks, times)[0])
    return c_index, c_index_ipcw, ibs, auc
