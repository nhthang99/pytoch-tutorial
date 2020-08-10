from valid import valid

import torch


def train(model,
            device,
            train_loader,
            valid_loader,
            optimizer,
            loss_fn,
            epoch=20):
    """Train

    Args:
        model (model): model architecture
        device (str): cuda or cpu
        train_loader (DataLoader): train data loader
        valid_loader (DataLoader): valid data loader
        optimizer (optim): optimizer
        loss_fn (loss): loss function
        epoch (int, optional): number of epochs. Defaults to 20.
    
    Returns:
        model: model's architecture and parameters
    """

    for idx_epoch in range(epoch):
        train_loss, train_acc = train_epoch(model, device,
                                            train_loader, optimizer,
                                            loss_fn)

        valid_loss, valid_acc = valid(model, device, valid_loader, loss_fn)

        print("Epoch {}:\tTrain loss: {:.3f}\tTrain accuracy: {:.3f}\tTrain loss: {:.3f}\tTrain accuracy: {:.3f}"
                    .format(idx_epoch, train_loss, train_acc, valid_loss, valid_acc))

    return model

def train_epoch(model,
                device,
                train_loader,
                optimizer,
                loss_fn):
    """Training for each epoch

    Args:
        model (model): model architecture
        device (str): cuda or cpu
        train_loader (DataLoader): train data loader
        optimizer (optim): optimizer
        loss_fn (loss): loss function
    
    Return:
        tuple(float, float): train epoch loss and train epoch accuracy
    """
    epoch_loss = []
    epoch_acc = []
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        step_loss, step_acc = train_step(model, images, labels,
                                            optimizer, loss_fn)

        epoch_loss.append(step_loss / images.size()[0])
        epoch_acc.append(step_acc / images.size()[0])

    loss = sum(epoch_loss) / len(epoch_loss)
    acc = sum(epoch_acc) / len(epoch_acc)

    return loss, acc


def train_step(model,
                images,
                labels,
                optimizer,
                loss_fn):
    """Training step

    Args:
        model (model): model architecture
        images (tensor): images
        labels (tensor): labels
        optimizer (optim): optimizer
        loss_fn (loss): loss function
    
    Return:
        tuple(float, float): step loss and step accuracy
    """
    # Zero to gradients
    optimizer.zero_grad()
    # Prediction
    output = model(images)
    # Loss
    loss = loss_fn(output, labels)
    # Backward
    loss.backward()
    # Update weight
    optimizer.step()

    # Cacl loss and accuracy
    step_loss = loss.item()
    _, preds = torch.max(output, dim=1)
    step_acc = (preds == labels).sum().item()

    return step_loss, step_acc
        