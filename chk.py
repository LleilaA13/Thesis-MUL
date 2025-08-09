import torch
ckpt = torch.load("models/inceptionv3_cat_forgetting/RLcheckpoint.pth.tar")
print(ckpt.keys())

python src/classification/main_random.py   --resume   --model_path models/inceptionv3_cat_forgetting/with_0.3.pt/RLcheckpoint.pth.tar   --subset_indices_pa
th cat_forget_indices.pt   --val_y_file labels/val_ys.pth   --train_y_file labels/train_ys.pth   --mask_path dummy.pt   --arch inceptionv3   --dataset imagenet_zeus   --batch_size 64   --gpu 0   --save_dir models/
inceptionv3_cat_forgetting/eval_only