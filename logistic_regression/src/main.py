import argparse

import dataset
import model
import train

import torch


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--batch-size', type=int, default=64, help="Batch size")
    parse.add_argument('--lr', type=float, default=0.005, help="Learning rate")
    parse.add_argument('--momentum', type=float, default=0.9, help="Momentum")
    parse.add_argument('--epoch', type=int, default=50, help="Number of epochs")
    parse.add_argument('--cpu', action='store_true', help="Use CPU")
    parse.add_argument('--save', action='store_true', help="Save checkpoint")
    opt = parse.parse_args()

    # Load dataset
    train_loader, valid_loader = dataset.load_dataset(bs=opt.batch_size, transform=True)
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() and not opt.cpu else "cpu")
    model, optimizer, loss_fn = model.load_model(device, opt.lr, opt.momentum)

    model = train.train(model, device, train_loader, valid_loader, optimizer, loss_fn, opt.epoch)

    if opt.save:
        torch.save(model.state_dict(), "weights/w1.pt")