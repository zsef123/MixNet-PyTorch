from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader

from torchvision import transforms as T


def get_dataset(root, dtype="cifar10", resl=224):
    tr = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    if dtype == "mnist":
        dset = MNIST
    elif dtype == "cifar10":
        dset = CIFAR10
    elif dtype == "cifar100":
        dset = CIFAR100
    elif dtype == "imagenet":
        return imagenet(root, resl)

    train = dset(root, True,  transform=tr, download=True)
    valid = dset(root, False, transform=tr)
    return train, valid


def imagenet(root, resl):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    train = ImageFolder(
        root + "/train",
        T.Compose([
            T.Resize([resl, resl]),
            T.RandomResizedCrop(resl),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
    )

    valid = ImageFolder(
        root + "/val",
        T.Compose([
            T.Resize([resl, resl]),
            T.ToTensor(),
            normalize,
        ])
    )

    return train, valid


def get_loaders(root, batch_size, num_workers=32, dtype="cifar10", resl=224):
    train, valid = get_dataset(root, dtype, resl)
    
    train_loader = DataLoader(train,
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(valid,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
    