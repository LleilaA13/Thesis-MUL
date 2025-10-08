# 📝 Thesis Project TODO List

## ✅ Initial Setup
- [x] Fork and clone `Unlearn-Saliency` repo
- [x] Create folder structure (`notebooks/`, `src/`, `models/`, etc.)
- [x] Create and activate `conda` environment
- [x] Write and save `environment.yml`
## 🔍 Experiments
- [ ] Clean up `01_baseline_visualizations.ipynb`
- [ ] Visualize and save filters (pre-unlearning)
- [ ] Write `src/visualization.py` with reusable functions
- [ ] Train model on CIFAR-10 subset
- [ ] Apply salUN to remove one class
- [ ] Save both models to `/models/`

## 🎨 Post-Unlearning
- [ ] Create `03_post_unlearning_visualizations.ipynb`
- [ ] Visualize same neurons after unlearning
- [ ] Save outputs to `/assets/visualizations/post_unlearning/`

## 📘 Final Touches
- [ ] Update `README.md` with structure + instructions
- [ ] Save all outputs/plots
- [ ] Draft initial thesis in `/reports/`


### AFTER TRAINING:
	- python scripts/learn_unlearn.py
	-cp results/0model_SA_best.pth.tar models/
	-cp results/unlearned_model.pth models/

	
git add models/
git commit -m "Save trained and unlearned model"
git push
