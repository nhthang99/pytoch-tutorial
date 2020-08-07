import torch


def valid(model,
            device,
            valid_loader,
            loss_fn):
    """Valid model

    Args:
        model (model): model architecture
        device (str): cuda or cpu
        valid_loader (DataLoader): valid data loader
        loss_fn (loss): loss function

    Returns:
        tuple(float, float): valid loss and valid accuracy
    """
    loss = []
    correct = []
    for idx, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        labels = labels.to(device)

        loss_step, correct_step = valid_step(model, images,
                                                labels, loss_fn)
        
        loss.append(loss_step / images.size()[0])
        correct.append(correct_step / images.size()[0])
    
    valid_loss = sum(loss) / len(loss)
    valid_acc = sum(correct) / len(correct)

    return valid_loss, valid_acc

def valid_step(model,
                images,
                labels,
                loss_fn):
    """Valid step

    Args:
        model (model): model architecture
        images (tensor): images
        labels (tensor): labels
        loss_fn (loss): loss function

    Returns:
        [type]: [description]
    """
    output = model(images)

    loss = loss_fn(output, labels)

    _, preds = torch.max(output, dim=1)

    loss_step = loss.item()
    correct_step = (preds == labels).sum().item()

    return loss_step, correct_step
