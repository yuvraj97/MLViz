from typing import Union, Dict
import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np


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
    n, d = X.shape

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = torch.as_tensor(X).float().to(device)
    y = torch.as_tensor(y).float().to(device)

    prev_loss = np.inf
    model = nn.Sequential(nn.Linear(d, 1)).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        outputs = model(X)

        loss = criterion(outputs, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():

            params = model.state_dict()
            keys = list(params.keys())
            if "weight" in keys[0]:
                weight = params[keys[0]].cpu().numpy().reshape(d, 1)
                bias = params[keys[1]].cpu().numpy().reshape(1, 1)
            else:
                weight = params[keys[1]].cpu().numpy().reshape(d, 1)
                bias = params[keys[2]].cpu().numpy().reshape(1, 1)
            theta = np.vstack((bias, weight))

            if 0 <= prev_loss - loss.item() <= inputs["epsilon"]:
                break
            prev_loss = loss.item()
            yield theta, loss.item()

    with torch.no_grad():
        params = model.state_dict()
        keys = list(params.keys())
        if "weight" in keys[0]:
            weight = params[keys[0]].cpu().numpy().reshape(d, 1)
            bias = params[keys[1]].cpu().numpy().reshape(1, 1)
        else:
            weight = params[keys[1]].cpu().numpy().reshape(d, 1)
            bias = params[keys[2]].cpu().numpy().reshape(1, 1)
        return np.vstack((bias, weight))
