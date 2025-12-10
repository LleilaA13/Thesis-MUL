# Unified Lucent Commands for Cross-Experiment Comparison

import matplotlib.pyplot as plt
from lucent.optvis import render

# Define high-resolution, high-quality rendering parameters
from lucent.optvis import param, transform, objectives
IMG_SIZE = 300
transforms = [
    transform.pad(16, mode='constant', constant_value=.5),
    transform.jitter(8),
    transform.random_scale([1 + (i - 5) / 50. for i in range(11)]),
    transform.random_rotate(list(range(-10, 11)) + 5 * [0]),
    transform.jitter(4),
    transform.crop_or_pad_to(IMG_SIZE, IMG_SIZE)
]
param_f = lambda: param.image(IMG_SIZE, batch=1, decorrelate=True)
# models = {'Baseline': model_baseline, '10% Forget': model_10, ...}

target_to_compare = 'layer3_5_conv3:666'


# Plot one target across all models to see the trend
fig, axes = plt.subplots(1, len(models), figsize=(20, 5))
fig.suptitle(f'Visualization for: {target_to_compare}', fontsize=16)
for ax, (name, model) in zip(axes, models.items()):
    img = render.render_vis(model, target_to_compare, param_f=param_f, transforms=transforms, thresholds=(2048,))
    ax.imshow(img[0][0])
    ax.set_title(name)
    ax.axis('off')
plt.show()
