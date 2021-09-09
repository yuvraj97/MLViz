import torch
import torch.nn as nn
from torch.optim import SGD
import numpy as np
from torch import Tensor


class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def run(X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        epochs: int = 100,
        epsilon: float = 0.001):
    """
    :param np.ndarray X: n x d matrix
    :param np.ndarray y: d x 1 matrix
    :param float learning_rate: (default: 0.01)
    :param int epochs: (default: 100)
    :param int epsilon: (default: 0.001)
    :return np.ndarray theta: d x 1 matrix
    """

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    n, d = X.shape
    X: Tensor = torch.as_tensor(X).float().to(device)
    y: Tensor = torch.as_tensor(y).float().to(device)

    prev_loss = np.inf
    model = LinearRegression(d, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        outputs = model(X)

        loss = criterion(outputs, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if epoch % 50 == 0:
            print(f"epoch: {epoch}, loss: {loss.item()}")

        with torch.no_grad():
            if 0 <= prev_loss - loss.item() <= epsilon:
                break
            prev_loss = loss.item()

    with torch.no_grad():
        params = model.state_dict()
        return np.vstack(
            (
                params["linear.bias"].cpu().numpy(),
                params["linear.weight"].cpu().numpy().reshape(d, 1)
            )
        )
