import argparse

from utils.datasets import load_dataset
from model.model import Net
from model.train import train

import torch
import torch.optim as optim
import torch.nn as nn


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--train-csv-path", type=str, default="data/train.csv", help="Training csv path for label file")
    parse.add_argument("--valid-csv-path", type=str, default="data/valid.csv", help="Validation csv path for label file")
    parse.add_argument("--test-csv-path", type=str, default="data/test.csv", help="Testing csv path for label file")
    parse.add_argument("--batch-size", type=int, default=64, help="Batch size of images")
    parse.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parse.add_argument("--momentum", type=float, default=0.9, help="Momentum")
    parse.add_argument("--gamma", type=float, default=0.8, metavar="M",
                        help="Learning rate step gamma (default: 0.7)")
    opt = parse.parse_args()

    train_loader, valid_loader, test_loader = load_dataset(train_csv_path=opt.train_csv_path,
                                                            valid_csv_path=opt.valid_csv_path,
                                                            test_csv_path=opt.test_csv_path,
                                                            bs=opt.batch_size, workers=2, transform=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = Net()
    model.to(device)

    # Define hyperparameter, optimizer, loss, scheduler
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)
    loss_fn = nn.CrossEntropyLoss()
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=opt.gamma)

    train(model, device, train_loader, valid_loader, loss_fn, optimizer, epoch=50)