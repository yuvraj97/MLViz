from typing import Union, Dict
import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np


class LogisticRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def run(inputs: Dict[str, Union[str, int, float, np.ndarray]]):
    """
    :param inputs: Dict[str, Union[str, int, float]] (template): {
        "X": numpy.ndarray,  # n x d
        "y": numpy.ndarray,  # n x 1
        "lr": float,
        "epochs": int,
        "epsilon": float,
    }

    :return theta numpy.ndarray  # d x 1 matrix
    """

    X, y, learning_rate, epochs = inputs["X"], inputs["y"], inputs["lr"], inputs["epochs"]
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_samples, n_features = X.shape
    X = torch.as_tensor(X).float().to(device)
    y = torch.as_tensor(y).float().to(device)
    y = y.view(n_samples, 1)

    prev_loss = np.inf
    model = LogisticRegression(n_features, 1).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():

            if 0 <= prev_loss - loss.item() <= inputs["epsilon"]:
                break
            prev_loss = loss.item()

            params = list(model.parameters())
            params_L = [params[-1].item()]
            params_L.extend(
                [parameter.item() for parameter in params[0][0]]
            )
            theta = np.array(params_L).reshape((n_features + 1, 1))
            print(loss.item())
            yield theta, loss.item()

    with torch.no_grad():
        params = list(model.parameters())
        params_L = [params[-1].item()]
        params_L.extend(
            [parameter.item() for parameter in params[0][0]]
        )
        return np.array(params_L).reshape((n_features + 1, 1))
