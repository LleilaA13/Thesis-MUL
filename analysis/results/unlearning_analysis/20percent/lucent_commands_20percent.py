# Lucent Commands for the 20percent Experiment

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
targets = ['layer3_5_conv3:359', 'layer3_5_conv3:666', 'layer3_1_conv3:376', 'layer3_0_conv3:501', 'layer3_0_conv3:226', 'layer3_1_conv3:877', 'layer1_2_conv3:135', 'layer3_3_conv3:140', 'layer4_0_conv3:1116', 'layer2_1_conv3:172', 'layer3_2_conv3:366', 'layer3_4_conv3:558', 'layer3_4_conv3:59', 'layer2_0_conv1:92', 'layer2_1_conv3:466']


# Visualize top 15 targets for this experiment
fig, axes = plt.subplots(3, 5, figsize=(25, 15))
fig.suptitle(f'Top 15 Most Affected Features for the {exp_key} Experiment', fontsize=18)
for ax, target in zip(axes.flatten(), targets):
    img = render.render_vis(unlearned_model, target, param_f=param_f, transforms=transforms, thresholds=(2048,))
    ax.imshow(img[0][0])
    ax.set_title(target, fontsize=10)
    ax.axis('off')
for i in range(len(targets), len(axes.flatten())):
    axes.flatten()[i].axis('off')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
