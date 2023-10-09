import logging
import os
import random
from typing import Tuple
import ray

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data import get_split_data
from src.utils import evaluate
from src.models.train import train_model, __MODELS__

INIT_SEED = 42
RUNS = 30

VAL_SPLIT = 0.2
TEST_SPLIT = 0.2

HIDDEN_FEATURES = 32
SPLITS = [5, 10, 50, 100, 500, 1000]
BATCH_SIZE = 128
EPOCHS = 300
PATIENCE = 10

MODELS = __MODELS__
DATASETS = ["whas500", "GBSG2", "metabric", "flamby"]
MODES = ["step_bwd", "step_fwd", "linear", "pce", "spline"]


def test_model(args: Tuple[str, str, int, int]) -> pd.DataFrame:
    start_time = pd.Timestamp.now()
    dataset_name, model_name, splits, random_state = args
    res = {
        "seed": [random_state] * len(MODES),
        "dataset": [dataset_name] * len(MODES),
        "model": [model_name] * len(MODES),
        "splits": [splits] * len(MODES),
        "mode": [],
        "cid": [],
        "ciw": [],
        "ibs": [],
        "auc": [],
    }
    X_train, y_train, X_val, y_val, X_test, y_test = get_split_data(dataset_name, VAL_SPLIT, TEST_SPLIT, random_state)
    _, model = train_model(model_name, X_train, y_train, X_val, y_val, HIDDEN_FEATURES, splits, BATCH_SIZE, EPOCHS,
                           PATIENCE)
    for mode in MODES:
        metrics = evaluate(model, y_train, X_test, y_test, mode)
        res["mode"].append(mode)
        res["cid"].append(metrics[0])
        res["ciw"].append(metrics[1])
        res["ibs"].append(metrics[2])
        res["auc"].append(metrics[3])
    logging.info(f"Finished {dataset_name} {model_name} in {pd.Timestamp.now() - start_time}")
    return pd.DataFrame(res)


def run_experiments():
    random.seed(INIT_SEED)
    res = pd.DataFrame()
    bar = tqdm(range(RUNS))
    for i in bar:
        random_state = random.randint(0, 2**16)
        np.random.seed(random_state)
        for dataset_name in DATASETS:
            for model_name in MODELS:
                for splits in SPLITS:
                    bar.set_description(f"Run {i + 1} - {dataset_name} - {model_name} - {splits}")
                    _res = test_model((dataset_name, model_name, splits, random_state))
                    res = pd.concat([res, _res])
        res.to_csv(f"results/results_run_{i + 1}.csv")
        if os.path.exists(f"results/results_run_{i}.csv"):
            os.remove(f"results/results_run_{i}.csv")


ray.init()


@ray.remote
def run_experiment(dataset_name, model_name, splits, random_state):
    _res = test_model((dataset_name, model_name, splits, random_state))
    return _res


def run_experiments_parallel():
    random.seed(INIT_SEED)
    res = pd.DataFrame()
    bar = tqdm(range(RUNS))

    for i in bar:
        random_state = random.randint(0, 2 ** 16)
        np.random.seed(random_state)

        futures = []
        for dataset_name in DATASETS:
            for model_name in MODELS:
                for splits in SPLITS:
                    bar.set_description(f"Run {i + 1} - {dataset_name} - {model_name} - {splits}")
                    future = run_experiment.remote(dataset_name, model_name, splits, random_state)
                    futures.append(future)

        ray.wait(futures, num_returns=len(futures))
        results = ray.get(futures)
        for _res in results:
            res = pd.concat([res, _res])

        res.to_csv(f"results/results_run_{i + 1}.csv")
        if os.path.exists(f"results/results_run_{i}.csv"):
            os.remove(f"results/results_run_{i}.csv")


if __name__ == "__main__":
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=RuntimeWarning)
    run_experiments_parallel()
