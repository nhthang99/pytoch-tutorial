from valid import valid

import torch
import torch.nn.functional as F


def train(model,
            device,
            train_dataloader,
            valid_loader,
            loss_fn,
            optimizer,
            scheduler=None,
            epoch=100):
    """Train model

    Args:
        model (model): model architecture
        device (str): cuda or cpu
        train_dataloader (DataLoader): train data loader
        valid_loader (DataLoader): valid data loader
        loss_fn (loss): loss function
        optimizer (optim): optimizer for model
        scheduler (scheduler, optional): scheduler for change learning rate. Defaults to None.
        epoch (int, optional): number of epoch for training. Defaults to 100.
    """
    for idx in range(epoch):
        train_loss, train_acc = train_epoch(model, device, train_dataloader,
                                            loss_fn, optimizer, scheduler)
        
        valid_loss, valid_acc = valid(model, device,
                                        valid_loader, loss_fn)

        print("Epoch {}:\tTrain loss: {:.5f}\tTrain accuracy: {:.5f} \tValid loss: {:.5f}\tValid accuracy: {:.5f}"
                        .format(idx, train_loss, train_acc, valid_loss, valid_acc))

def train_epoch(model,
                device,
                train_dataloader,
                loss_fn,
                optimizer,
                scheduler):
    """Train model for each epoch

    Args:
        model (model): model architecture
        device (str): cuda or cpu
        train_dataloader (DataLoader): train data loader
        loss_fn (loss): loss function
        optimizer (optim): optimizer for model
        scheduler (scheduler, optional): scheduler for change learning rate. Defaults to None.

    Returns:
        tuple(float, float): training loss and training accuracy
    """
    loss = []
    correct = []
    # Training epoch
    for idx, (images, labels) in enumerate(train_dataloader):
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # Training step
        loss_step, correct_step = train_step(model, images,
                                            labels, loss_fn,
                                            optimizer, scheduler)
        
        # Calc loss and accuracy
        loss.append(loss_step / images.size()[0])
        correct.append(correct_step / images.size()[0])

    train_loss = sum(loss) / len(loss)
    train_acc = sum(correct) / len(correct)

    return train_loss, train_acc
        

def train_step(model,
                images,
                labels,
                loss_fn,
                optimizer,
                scheduler):
    """Training step

    Args:
        model (model): model architecture
        images (tensor): images
        labels (tensor): labels
        loss_fn (loss): loss function
        optimizer (optim): optimizer for model
        scheduler (scheduler, optional): scheduler for change learning rate. Defaults to None.

    Returns:
        tuple(float, int): training step loss and training step correct
    """
    # Zero the gradients before running the backward pass.
    optimizer.zero_grad()
    # Prediction
    output = model(images)
    # Calc loss
    loss = loss_fn(output, labels)
    # Backward
    loss.backward()
    # Update weight
    optimizer.step()
    # Scheduler
    if scheduler:
        scheduler.step()
    # Calc loss and correct step
    loss_step = loss.item()
    _, preds = torch.max(output, dim=1)
    correct_step = (preds == labels).sum().item()

    return loss_step, correct_step
