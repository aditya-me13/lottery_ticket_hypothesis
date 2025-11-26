import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(dataset="mnist", batch_size=64, data_dir="./data"):
    # pick transforms and dataset class
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = datasets.MNIST
        channels = 1

    elif dataset == "fashion":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_set = datasets.FashionMNIST
        channels = 1

    elif dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
        ])
        train_set = datasets.CIFAR10
        channels = 3
    else:
        raise ValueError("dataset must be: 'mnist', 'fashion', or 'cifar10'")

    # load datasets
    if dataset == "cifar10":
        train_data = train_set(root=data_dir, train=True, download=True, transform=transform_train)
        test_data = train_set(root=data_dir, train=False, download=True, transform=transform_test)
        train_size, val_size = 45000, 5000
    else:
        train_data = train_set(root=data_dir, train=True, download=True, transform=transform)
        test_data = train_set(root=data_dir, train=False, download=True, transform=transform)
        train_size, val_size = 55000, 5000

    # train/val split
    train_data, val_data = random_split(
        train_data,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader
