# Real Influence Analysis - Lucent Visualization Commands
# Copy these into Google Colab

from lucent.optvis import render
import matplotlib.pyplot as plt

# Top individual targets
# 1. Channel 387 in layer2.2.conv3 (Change: 1617.4023)
img_1 = render.render_vis(model, "layer2.2.conv3:387", show_inline=True, thresholds=(512,))

# 2. Channel 53 in layer1.2.conv3 (Change: 791.7326)
img_2 = render.render_vis(model, "layer1.2.conv3:53", show_inline=True, thresholds=(512,))

# 3. Channel 424 in layer4.0.conv3 (Change: 755.0659)
img_3 = render.render_vis(model, "layer4.0.conv3:424", show_inline=True, thresholds=(512,))

# 4. Channel 482 in layer3.5.conv3 (Change: 586.0610)
img_4 = render.render_vis(model, "layer3.5.conv3:482", show_inline=True, thresholds=(512,))

# 5. Channel 41 in layer1.2.conv1 (Change: 460.2436)
img_5 = render.render_vis(model, "layer1.2.conv1:41", show_inline=True, thresholds=(512,))

# 6. Channel 375 in layer4.0.conv3 (Change: 430.2732)
img_6 = render.render_vis(model, "layer4.0.conv3:375", show_inline=True, thresholds=(512,))

# 7. Channel 32 in layer4.0.conv3 (Change: 366.6627)
img_7 = render.render_vis(model, "layer4.0.conv3:32", show_inline=True, thresholds=(512,))

# 8. Channel 67 in layer3.5.conv3 (Change: 353.6905)
img_8 = render.render_vis(model, "layer3.5.conv3:67", show_inline=True, thresholds=(512,))

# 9. Channel 53 in layer2.2.conv2 (Change: 232.6943)
img_9 = render.render_vis(model, "layer2.2.conv2:53", show_inline=True, thresholds=(512,))

# 10. Channel 1270 in layer4.0.conv3 (Change: 219.8087)
img_10 = render.render_vis(model, "layer4.0.conv3:1270", show_inline=True, thresholds=(512,))


# Batch visualization of top 6
targets = [
    "layer2.2.conv3:387",  # Channel 387 in layer2.2.conv3 (Change: 1617.4023)
    "layer1.2.conv3:53",  # Channel 53 in layer1.2.conv3 (Change: 791.7326)
    "layer4.0.conv3:424",  # Channel 424 in layer4.0.conv3 (Change: 755.0659)
    "layer3.5.conv3:482",  # Channel 482 in layer3.5.conv3 (Change: 586.0610)
    "layer1.2.conv1:41",  # Channel 41 in layer1.2.conv1 (Change: 460.2436)
    "layer4.0.conv3:375",  # Channel 375 in layer4.0.conv3 (Change: 430.2732)
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, target in enumerate(targets):
    img = render.render_vis(model, target, show_inline=False, thresholds=(256,))
    if hasattr(img, 'cpu'):
        img_np = img.cpu().numpy().transpose(1, 2, 0)
    else:
        img_np = np.array(img)
    axes[i].imshow(img_np)
    axes[i].set_title(target, fontsize=10)
    axes[i].axis('off')

plt.suptitle('Real Analysis: Top 6 Most Influenced Components', fontsize=16)
plt.tight_layout()
plt.show()
