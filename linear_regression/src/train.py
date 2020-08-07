import torch


def train(model,
            optimizer,
            X_train,
            y_train,
            loss_fn,
            epoch=20):
    """Train model

    Args:
        model (model): model architecture
        optimizer (optim): optimizer for model
        X_train (tensor): data
        y_train (tensor): label
        loss_fn ([type]): loss function
        epoch (int, optional): number of epoch for training. Defaults to 20.

    Returns:
        model: model
    """
    # Cast tensor to float
    X_train = X_train.float()
    y_train = y_train.float()

    for idx_epoch in range(epoch):
        train_loss, train_acc = train_epoch(model, optimizer,
                                                X_train, y_train, loss_fn)

        print("Epoch {}:\tTrain loss: {:.4f}\tTrain acc: {:.4f}\n"
                    .format(idx_epoch, train_loss, train_acc))
    return model


def train_epoch(model,
                optimizer,
                X_train,
                y_train,
                loss_fn):
    """Train model for each epoch

    Args:
        model (model): model architecture
        optimizer (optim): optimizer for model
        X_train (tensor): data
        y_train (tensor): label
        loss_fn ([type]): loss function

    Returns:
        tuple(float, float): loss and accuracy for each epoch
    """
    # Reset to zero gradients
    optimizer.zero_grad()
    # Prediction
    output = model(X_train)
    # Loss
    loss = loss_fn(output, y_train)
    # Backwark
    loss.backward()
    # Update weight
    optimizer.step()

    # Loss for each epoch
    loss_epoch = loss.item()
    loss_epoch = loss_epoch / X_train.size()[0]

    # Accuracy
    correct = (torch.abs(output - y_train) < 0.1).sum().item()
    acc = correct / X_train.size()[0]
    
    return loss_epoch, acc