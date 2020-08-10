from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_dataset(bs, transform=True):
    """Load MNIST dataset

    Args:
        bs (int): batch size
        transform (bool, optional): transform dataset. Defaults to True.

    Returns:
        tuple(DataLoader, DataLoader): train and test data loader
    """
    trans = None
    # Transforms dataset
    if transform:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5,], [0.5])
        ])
    # Load dataset
    train_dataset = datasets.MNIST(root='data/', train=True, transform=trans, download=True)
    test_dataset = datasets.MNIST(root='data/', train=False, transform=trans, download=True)

    # Load data using Data Loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=bs, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs, num_workers=1)

    return train_loader, test_loader
