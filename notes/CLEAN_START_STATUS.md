# Clean Start Status Report

## What We're Keeping (InceptionV3 Related)
- ✅ `models/inceptionv3_cat_forgetting/` - InceptionV3 experiments from August
- ✅ `masks/inceptionv3_cat_forgetting/` - InceptionV3 mask files (working)

## What We've Cleaned Up
- ❌ Renamed corrupted cat files:
  - `cat_forget_indices_OLD_CORRUPTED.pt` (1.2M samples, wrong format)
  - `cats_forget_indices_OLD_CORRUPTED.pt` (100K samples, wrong format)

## Fresh Start Files Created
- ✅ `dogs_forget_indices.pt` - 6 dog classes, 3000 samples (500 per class)
- ✅ `vehicles_forget_indices_CLEAN.pt` - 7 vehicle classes, 3500 samples (500 per class)

## Dog Classes for Experiments (6 classes, 3000 samples):
- Index 11: German shepherd, German shepherd dog, German police dog, alsatian
- Index 39: Labrador retriever
- Index 78: golden retriever  
- Index 135: Yorkshire terrier
- Index 182: Chihuahua
- Index 194: standard poodle

## Vehicle Classes for Experiments (7 classes, 3500 samples):
- Index 15: school bus
- Index 52: freight car
- Index 64: moving van
- Index 90: police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria
- Index 117: sports car, sport car
- Index 147: beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon
- Index 152: trolleybus, trolley coach, trackless trolley

## Next Steps:
1. Start with dog forgetting experiments (highly interpretable for Lucent)
2. Then move to vehicle experiments
3. Keep InceptionV3 cat experiments as reference/comparison

## Why Dogs First:
- Most interpretable features (facial features, fur, ears, etc.)
- Clear semantic meaning
- Good variety (6 different breeds)
- Excellent for feature visualization and thesis presentation