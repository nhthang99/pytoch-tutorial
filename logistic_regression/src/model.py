import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Logistic(nn.Module):
    """Logistic model
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = F.softmax(self.fc(x), dim=1)
        return x


def load_model(device, lr, momentum):
    """Load model

    Args:
        device (str): cuda or cpu
        lr (float): learning rate
        momentum (float): momentum

    Returns:
        tuple(model, optim, loss_fn): model architecture, optimizer and loss function
    """
    model = Logistic()
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    loss_fn = nn.CrossEntropyLoss()

    return model, optimizer, loss_fn