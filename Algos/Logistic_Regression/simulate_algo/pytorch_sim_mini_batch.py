from typing import Union, Dict
import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np
from torch import Tensor


class LogisticRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


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

    dataset = Dataset(X, y)
    n_samples, n_features = X.shape
    X = torch.as_tensor(X).float().to(dataset.device)
    y = torch.as_tensor(y).float().to(dataset.device)
    y = y.view(n_samples, 1)

    prev_loss = np.inf
    model = LogisticRegression(n_features, 1).to(dataset.device)
    criterion = nn.BCELoss().to(dataset.device)
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
            if 0 <= prev_loss - loss.item() <= inputs["epsilon"]:
                break
            prev_loss = loss.item()

            params = list(model.parameters())
            params_L = [params[-1].item()]
            params_L.extend(
                [parameter.item() for parameter in params[0][0]]
            )

            yield np.array(params_L).reshape((n_features + 1, 1)), loss.item()

    with torch.no_grad():
        params = list(model.parameters())
        params_L = [params[-1].item()]
        params_L.extend(
            [parameter.item() for parameter in params[0][0]]
        )
        return np.array(params_L).reshape((n_features + 1, 1))  # d x 1
