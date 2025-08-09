import torch
from torchvision import transforms
from imagenet import prepare_data

loaders = prepare_data(
    dataset="imagenet_zeus",
    batch_size=1,
    shuffle=False,
    data_path="/media/pinas/datasets/imagenet_zeus"
)

val_loader = loaders["val"]

# Force transform deep into dataset
resize_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

target_dataset = val_loader.dataset
while hasattr(target_dataset, "dataset"):
    target_dataset = target_dataset.dataset

target_dataset.transform = resize_transform

# Now print image sizes
for i, (img, label) in enumerate(val_loader):
    print(f"[{i}] Image shape: {img.shape}")
    if i > 10:
        break
