from typing import Union, Dict
import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np
from torch import Tensor

from Algos.utils.utils import get_MSE_error


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.X: Tensor = torch.as_tensor(X).float().to(self.device)
        self.y: Tensor = torch.as_tensor(y).float().to(self.device)

    def get_batches(self, batch_size: int):
        n, d = self.X.shape
        n_rounds = n // batch_size
        for round_i in range(n_rounds + 1):
            yield [
                self.X[round_i * batch_size: (round_i + 1) * batch_size],
                self.y[round_i * batch_size: (round_i + 1) * batch_size]
            ]


def run(inputs: Dict[str, Union[str, int, float, np.ndarray]]):
    """
    :param inputs: Dict[str, Union[str, int, float]] (template): {
        "X": numpy.ndarray,  # n x d
        "y": numpy.ndarray,  # n x 1
        "lr": float,
        "epochs": int,
        "epsilon": float,
        "batch_size": int
    }

    :return theta numpy.ndarray  # d x 1 matrix
    """

    X, y, learning_rate = inputs["X"], inputs["y"], inputs["lr"]
    epochs, batch_size = inputs["epochs"], inputs["batch_size"]

    n, d = X.shape

    dataset = Dataset(X, y)
    X = torch.from_numpy(X).float().to(dataset.device)
    y = torch.from_numpy(y).float().to(dataset.device)
    prev_loss = np.inf
    model = nn.Sequential(nn.Linear(d, 1)).to(dataset.device)
    criterion = nn.MSELoss().to(dataset.device)
    optimizer = SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for batch_i, (X_batch, y_batch) in enumerate(dataset.get_batches(batch_size)):

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            outputs = model(X)
            loss = criterion(outputs, y)

            params = model.state_dict()
            keys = list(params.keys())
            theta = np.vstack(
                (params[keys[1]].cpu().numpy(), params[keys[0]].cpu().numpy())
            )

            if 0 <= prev_loss - loss.item() <= inputs["epsilon"]:
                return theta
            prev_loss = loss.item()

            yield theta, loss.item()

    with torch.no_grad():
        params = model.state_dict()
        keys = list(params.keys())
        return np.vstack(
            (params[keys[1]].cpu().numpy(), params[keys[0]].cpu().numpy())
        )
