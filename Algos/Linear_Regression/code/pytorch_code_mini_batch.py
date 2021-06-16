import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np
from torch import Tensor


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

def run(X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 100,
        epsilon: float = 0.001,
        batch_size: int = 10):
    """
    :param np.ndarray X: n x d matrix
    :param np.ndarray y: d x 1 matrix
    :param float learning_rate: (default: 0.01)
    :param int epochs: (default: 100)
    :param int epsilon: (default: 0.001)
    :param int batch_size: (default: 10)
    :return np.ndarray theta: d x 1 matrix
    """

    n, d = X.shape

    dataset = Dataset(X, y)
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

            if batch_i + epoch * batch_size % 50 == 0:
                print(f"epoch: {batch_i + epoch * batch_size}, loss: {loss.item()}")

            with torch.no_grad():
                if 0 <= prev_loss - loss.item() <= epsilon:
                    params = model.state_dict()
                    keys = list(params.keys())
                    return np.vstack(
                        (params[keys[1]].cpu().numpy(), params[keys[0]].cpu().numpy())
                    )
                prev_loss = loss.item()

    with torch.no_grad():
        params = model.state_dict()
        keys = list(params.keys())
        return np.vstack(
            (params[keys[1]].cpu().numpy(), params[keys[0]].cpu().numpy())
        )
