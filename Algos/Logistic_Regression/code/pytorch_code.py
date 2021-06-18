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


def run(X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.1,
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
    n_samples, n_features = X.shape
    X: Tensor = torch.as_tensor(X).float().to(device)
    y: Tensor = torch.as_tensor(y).float().to(device)
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

        if epoch % 50 == 0:
            print(f"epoch: {epoch}, loss: {loss.item()}")

        with torch.no_grad():
            if 0 <= prev_loss - loss.item() <= epsilon:
                break
            prev_loss = loss.item()

    with torch.no_grad():
        params = list(model.parameters())
        params_L = [params[-1].item()]
        params_L.extend(
            [parameter.item() for parameter in params[0][0]]
        )
        return np.array(params_L).reshape((n_features + 1, 1))  # d x 1
