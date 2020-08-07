import argparse

from dataset import load_dataset
from train import train

import torch
import torch.nn as nn
import torch.optim as optim


def model_definition(in_features,
                        out_features,
                        lr,
                        momentum=0.9):
    """Model definition

    Args:
        in_features (int): number of input features
        out_features (int): number of output features
        lr (float): learning rate
        momentum (float): momentum to avoid local minimum

    Returns:
        tuple(model, loss_fn, optimizer): model configure and hyperparameter
    """
    model = nn.Linear(in_features, out_features)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    return model, loss_fn, optimizer

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--data', type=str, default="data/data.csv", help="Path to data")
    parse.add_argument('--lr', type=float, default=0.005, help="Learning rate")
    parse.add_argument('--momentum', type=float, default=0.9, help="Momentum for avoid local minimum")
    parse.add_argument('--save', action='store_true', help="Save checkpoint")
    opt = parse.parse_args()

    # Load dataset
    X_train, y_train = load_dataset(opt.data)

    # Model definition
    model, loss_fn, optimizer = model_definition(X_train.size()[1], y_train.size()[1], opt.lr, opt.momentum)

    # Train model
    model = train(model, optimizer, X_train, y_train, loss_fn, epoch=100)

    # Save checkpoint
    if opt.save:
        torch.save(model.state_dict(), "weights/linear.pt")
