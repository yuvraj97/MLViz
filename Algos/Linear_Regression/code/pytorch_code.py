import torch
import torch.nn as nn
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
        epochs: int = 100):

    """
    :param np.ndarray X: n x d matrix
    :param np.ndarray y: d x 1 matrix
    :param float learning_rate: (default: 0.01)
    :param int epochs: (default: 100)
    :return np.ndarray theta: d x 1 matrix
    """

    X: Tensor = torch.from_numpy(X.astype(np.float32))
    y: Tensor = torch.from_numpy(y.astype(np.float32))

    input_size = output_size = X.shape[1]
    model = LinearRegression(input_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        y_prediction = model(X)
        loss = criterion(y, y_prediction)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return [parameter.item() for parameter in model.parameters()]
