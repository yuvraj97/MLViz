import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import NeuralNet

def train(hidden_size1=100, hidden_size2=100, batch_size=100, learning_rate=0.001, n_epochs=2):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    input_size = 28 * 28  # As our Image is 28 * 28
    num_classes = 10

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(),
                                               download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    model = NeuralNet(input_size, hidden_size1, hidden_size2, num_classes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(n_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('\r', f"{(_ * len(train_loader) + i) / (n_epochs * len(train_loader)) * 100:.2f}% completed", end="")
    print('\r', "100% completed")
    torch.save(model.state_dict(), "model.mnist.pytorch")
    return model


def test(hidden_size1=100, hidden_size2=100, batch_size=100):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    input_size = 28 * 28  # As our Image is 28 * 28
    num_classes = 10
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    model = NeuralNet(input_size, hidden_size1, hidden_size2, num_classes)
    model.load_state_dict(torch.load("model.mnist.pytorch"))
    model.eval()

    with torch.no_grad():
        n_correct, n_samples = 0, 0
        for images, labels in test_loader:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            values, indexes = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (indexes == labels).sum().item()
        accuracy = 100.0 * n_correct / n_samples
        return accuracy


# lr = 0.004375
# n_epochs = 3
# train(
#     hidden_size1=128,
#     hidden_size2=64,
#     batch_size=100,
#     learning_rate=lr,
#     n_epochs=n_epochs
# )
# accuracy = test(
#     hidden_size1=128,
#     hidden_size2=64,
#     batch_size=100,
# )
# print(f"LR: {lr}, n_epochs: {n_epochs} accuracy: {accuracy}")
