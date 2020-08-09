from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    """Custom dataset
    """

    def __init__(self, csv_path, trans=None):
        super(ImageDataset, self).__init__()

        self.df = pd.read_csv(csv_path, header=0)
        self.df["Label"] = self.df["Label"].astype(int)

        self.trans = trans

    def __getitem__(self, index):
        image = Image.open(self.df.iloc[index, 0]).convert('RGB')
        label = self.df.iloc[index, 1]
        
        if self.trans:
            image = self.trans(image)
        
        return image, label
    
    def __len__(self):
        return self.df.shape[0]


def load_dataset(train_csv_path,
                    valid_csv_path,
                    test_csv_path,
                    bs=32,
                    workers=4,
                    transform=True):
    """Load dataset

    Args:
        train_csv_path (str): training csv path for label file.
        valid_csv_path (str): validation csv path for label file.
        test_csv_path (str): test csv path for label file.
        bs (int): batch size of images. Defaults to 32.
        workers (int): number of workers for training loader. Defaults to 4.
        transform (bool, optional): Transform dataset. Defaults to True.

    Returns:
        tuple(DataLoader, DataLoader, DataLoader): Train-, valid- test-dataloader.
    """
    trans = None
    # Transform dataset
    if transform:
        trans = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomResizedCrop(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,],std=[0.229,])
        ])

    # Load dataset
    train_dataset = ImageDataset(train_csv_path, trans)
    valid_dataset = ImageDataset(valid_csv_path, trans)
    test_dataset = ImageDataset(test_csv_path, trans)

    # Load dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=bs, num_workers=workers, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=bs, num_workers=1, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs, num_workers=1, shuffle=False)

    return train_loader, valid_loader, test_loader
