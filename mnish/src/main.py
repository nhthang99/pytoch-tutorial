import argparse

from dataset import load_dataset
from model import Net
from train import train

import torch
import torch.optim as optim
import torch.nn as nn

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--batch-size', type=int, default=8, help="Batch size of images")
    parse.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parse.add_argument('--momentum', type=float, default=0.9, help="Momentum")
    parse.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    opt = parse.parse_args()

    train_dataloader, valid_dataloader = load_dataset(bs=64, transform=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = Net()
    model.to(device)

    # Define hyperparameter, optimizer, loss, scheduler
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)
    loss_fn = nn.NLLLoss()
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=opt.gamma)

    train(model, device, train_dataloader, valid_dataloader, loss_fn, optimizer, epoch=50)