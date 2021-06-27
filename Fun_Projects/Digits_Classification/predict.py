import torch
import numpy as np
from torch import Tensor
from Fun_Projects.Digits_Classification.model import NeuralNet


def predict(image):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    input_size = 28 * 28  # As our Image is 28 * 28
    model = NeuralNet(input_size, hidden_size1=128, hidden_size2=64, num_classes=10).to(device)
    model.load_state_dict(torch.load("Fun_Projects/Digits_Classification/model.mnist.pytorch"))
    model.eval()
    image = torch.Tensor(image).reshape(-1, input_size).to(device)
    output: Tensor = model(image)
    return np.array([i.item() for i in output[0]]).argsort()[::-1]
