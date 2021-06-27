import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(NeuralNet, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.linear1 = nn.Linear(input_size, hidden_size1).to(device)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2).to(device)
        self.linear3 = nn.Linear(hidden_size2, num_classes).to(device)
        self.activation_function = nn.ReLU().to(device)
        # Here we do not use SoftMax here because
        # We are using CrossEntropyLoss which itself uses SoftMax

    def forward(self, x):
        op1 = self.activation_function(self.linear1(x))
        op2 = self.activation_function(self.linear2(op1))
        op3 = self.linear3(op2)
        return op3
