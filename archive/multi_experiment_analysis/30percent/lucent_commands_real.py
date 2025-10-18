# Real Influence Analysis - Lucent Visualization Commands
# Copy these into Google Colab

from lucent.optvis import render
import matplotlib.pyplot as plt

# Top individual targets
# 1. Channel 274 in layer3.1.conv3 (Change: 3675.7275)
img_1 = render.render_vis(model, "layer3.1.conv3:274", show_inline=True, thresholds=(512,))

# 2. Channel 51 in layer3.0.conv1 (Change: 1786.9342)
img_2 = render.render_vis(model, "layer3.0.conv1:51", show_inline=True, thresholds=(512,))

# 3. Channel 44 in layer2.3.conv2 (Change: 1058.8596)
img_3 = render.render_vis(model, "layer2.3.conv2:44", show_inline=True, thresholds=(512,))

# 4. Channel 31 in layer1.0.conv3 (Change: 680.7674)
img_4 = render.render_vis(model, "layer1.0.conv3:31", show_inline=True, thresholds=(512,))

# 5. Channel 365 in layer4.1.conv2 (Change: 247.3791)
img_5 = render.render_vis(model, "layer4.1.conv2:365", show_inline=True, thresholds=(512,))

# 6. Channel 3 in layer3.3.conv2 (Change: 246.5996)
img_6 = render.render_vis(model, "layer3.3.conv2:3", show_inline=True, thresholds=(512,))

# 7. Channel 466 in layer4.0.conv2 (Change: 243.6949)
img_7 = render.render_vis(model, "layer4.0.conv2:466", show_inline=True, thresholds=(512,))

# 8. Channel 51 in layer2.1.conv2 (Change: 191.0499)
img_8 = render.render_vis(model, "layer2.1.conv2:51", show_inline=True, thresholds=(512,))

# 9. Channel 40 in layer4.1.conv2 (Change: 154.6919)
img_9 = render.render_vis(model, "layer4.1.conv2:40", show_inline=True, thresholds=(512,))

# 10. Channel 250 in layer3.0.conv2 (Change: 117.1454)
img_10 = render.render_vis(model, "layer3.0.conv2:250", show_inline=True, thresholds=(512,))


# Batch visualization of top 6
targets = [
    "layer3.1.conv3:274",  # Channel 274 in layer3.1.conv3 (Change: 3675.7275)
    "layer3.0.conv1:51",  # Channel 51 in layer3.0.conv1 (Change: 1786.9342)
    "layer2.3.conv2:44",  # Channel 44 in layer2.3.conv2 (Change: 1058.8596)
    "layer1.0.conv3:31",  # Channel 31 in layer1.0.conv3 (Change: 680.7674)
    "layer4.1.conv2:365",  # Channel 365 in layer4.1.conv2 (Change: 247.3791)
    "layer3.3.conv2:3",  # Channel 3 in layer3.3.conv2 (Change: 246.5996)
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
