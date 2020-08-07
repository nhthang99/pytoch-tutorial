from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_dataset(bs=8, transform=True):
    """Load MNIST dataset

    Args:
        bs (int): Batch size of images. Defaults to 8
        transform (bool, optional): Transform dataset. Defaults to True.

    Returns:
        DataLoader, DataLoader: Train and test data loader.
    """
    trans = None
    # Transform dataset
    if transform:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.0,], [1,])
        ])

    # Load dataset
    train_dataset = datasets.MNIST(root='data/', train=True, transform=trans, download=True)
    valid_dataset = datasets.MNIST(root='data/', train=False, transform=trans, download=True)

    # Load dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=bs, num_workers=2)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=bs, num_workers=1)

    return train_dataloader, valid_dataloader