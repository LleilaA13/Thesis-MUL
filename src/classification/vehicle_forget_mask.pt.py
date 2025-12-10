import torch

# Read WNIDs
with open("tiny-imagenet-200/wnids.txt") as f:
    wnids = [line.strip() for line in f]

vehicle_wnids = [
    "n02690373", "n02958343", "n02974003", "n03100240", "n03417042",
    "n03770679", "n03796401", "n03930630", "n04037443", "n04285008", "n04461696"
]
vehicle_indices = [wnids.index(w) for w in vehicle_wnids if w in wnids]

# Assuming train_labels is a list/array of class indices for your training set
mask = torch.tensor([1 if label in vehicle_indices else 0 for label in train_labels])
torch.save(mask, "vehicle_forget_mask.pt")
