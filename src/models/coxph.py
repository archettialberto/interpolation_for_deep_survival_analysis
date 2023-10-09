import torch
from pycox.models import CoxPH
import torchtuples as tt

from src.utils import to_pycox_labels


class Net(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.output = torch.nn.Linear(
            in_features=in_features, out_features=1, bias=False
        )

    def forward(self, x):
        x = self.output(x)
        return x


def train_cox(
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
    _y_train = to_pycox_labels(y_train)
    _y_val = to_pycox_labels(y_val)
    val_data = (X_val, _y_val)
    model = CoxPH(Net(X_train.shape[1]), torch.optim.Adam)
    callbacks = [tt.callbacks.EarlyStopping(patience=patience)]
    log = model.fit(
        X_train,
        _y_train,
        epochs=epochs,
        val_data=val_data,
        verbose=False,
        callbacks=callbacks,
        batch_size=batch_size,
    )
    model.compute_baseline_hazards(X_train, _y_train)
    return log, model
