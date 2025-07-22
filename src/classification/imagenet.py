import os
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def prepare_data(
    dataset,
    batch_size=512,
    shuffle=True,
    train_subset_indices=None,
    val_subset_indices=None,
    data_path="/media/pinas/datasets/imagenet_zeus",
):
    if dataset == "imagenet_zeus":
        train_dir = os.path.join(data_path, "train")
        val_dir = os.path.join(data_path, "val")

        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
        ])

        transform_val = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
        ])

        train_set = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
        validation_set = torchvision.datasets.ImageFolder(val_dir, transform=transform_val)

    else:
        raise NotImplementedError(f"Dataset {dataset} not supported.")

    # Subset logic
    if train_subset_indices is not None:
        forget_indices = torch.ones_like(train_subset_indices) - train_subset_indices
        train_subset_indices = torch.nonzero(train_subset_indices).squeeze()
        forget_indices = torch.nonzero(forget_indices).squeeze()

        retain_set = Subset(train_set, train_subset_indices)
        forget_set = Subset(train_set, forget_indices)
    if val_subset_indices is not None:
        val_subset_indices = torch.nonzero(val_subset_indices).squeeze()
        validation_set = Subset(validation_set, val_subset_indices)

    if train_subset_indices is not None:
        loaders = {
            "train": DataLoader(retain_set, batch_size=batch_size, num_workers=8, shuffle=shuffle),
            "val": DataLoader(validation_set, batch_size=batch_size, num_workers=8, shuffle=False),
            "fog": DataLoader(forget_set, batch_size=batch_size, num_workers=8, shuffle=False),
        }
    else:
        loaders = {
            "train": DataLoader(train_set, batch_size=batch_size, num_workers=8, shuffle=shuffle),
            "val": DataLoader(validation_set, batch_size=batch_size, num_workers=8, shuffle=False),
        }

    return loaders


def get_x_y_from_data_dict(data, device):
    x, y = data  # Data is a tuple: (image, label)
    return x.to(device), y.to(device)


if __name__ == "__main__":
    # Example: dump all labels for train/val
    ys = {"train": [], "val": []}
    loaders = prepare_data(
        dataset="imagenet_zeus",
        batch_size=1,
        shuffle=False,
        data_path="/media/pinas/datasets/imagenet_zeus",
    )

    for data in tqdm(loaders["train"], desc="Collecting train labels"):
        _, y = get_x_y_from_data_dict(data, "cpu")
        ys["train"].append(y.item())

    for data in tqdm(loaders["val"], desc="Collecting val labels"):
        _, y = get_x_y_from_data_dict(data, "cpu")
        ys["val"].append(y.item())

    torch.save(torch.tensor(ys["train"]).long(), "train_ys.pth")
    torch.save(torch.tensor(ys["val"]).long(), "val_ys.pth")
