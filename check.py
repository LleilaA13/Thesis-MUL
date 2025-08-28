import torch
ckpt = torch.load("models/inceptionv3_cat_forgetting/RLcheckpoint.pth.tar", map_location="cpu")
print(ckpt.keys() if isinstance(ckpt, dict) else type(ckpt))
if isinstance(ckpt, dict):
    print("eval result keys:", ckpt.get("evaluation_result", {}).keys())
    print("accuracies:", ckpt.get("evaluation_result", {}).get("accuracy"))
