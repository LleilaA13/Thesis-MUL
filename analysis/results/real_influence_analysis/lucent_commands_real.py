# Real Influence Analysis - Lucent Visualization Commands
# Copy these into Google Colab

from lucent.optvis import render
import matplotlib.pyplot as plt

# Top individual targets
# 1. Channel 125 in layer2.0.conv2 (Change: 874.1881)
img_1 = render.render_vis(model, "layer2.0.conv2:125", show_inline=True, thresholds=(512,))

# 2. Channel 297 in layer2.2.conv3 (Change: 832.5737)
img_2 = render.render_vis(model, "layer2.2.conv3:297", show_inline=True, thresholds=(512,))

# 3. Channel 222 in layer3.3.conv2 (Change: 795.9881)
img_3 = render.render_vis(model, "layer3.3.conv2:222", show_inline=True, thresholds=(512,))

# 4. Channel 33 in layer2.3.conv2 (Change: 709.2872)
img_4 = render.render_vis(model, "layer2.3.conv2:33", show_inline=True, thresholds=(512,))

# 5. Channel 0 in layer2.0.conv1 (Change: 352.0150)
img_5 = render.render_vis(model, "layer2.0.conv1:0", show_inline=True, thresholds=(512,))

# 6. Channel 213 in layer2.2.conv3 (Change: 204.4977)
img_6 = render.render_vis(model, "layer2.2.conv3:213", show_inline=True, thresholds=(512,))

# 7. Channel 103 in layer4.0.conv2 (Change: 193.8776)
img_7 = render.render_vis(model, "layer4.0.conv2:103", show_inline=True, thresholds=(512,))

# 8. Channel 110 in layer3.2.conv2 (Change: 154.8611)
img_8 = render.render_vis(model, "layer3.2.conv2:110", show_inline=True, thresholds=(512,))

# 9. Channel 332 in layer2.2.conv3 (Change: 143.1671)
img_9 = render.render_vis(model, "layer2.2.conv3:332", show_inline=True, thresholds=(512,))

# 10. Channel 246 in layer3.0.conv2 (Change: 128.9345)
img_10 = render.render_vis(model, "layer3.0.conv2:246", show_inline=True, thresholds=(512,))


# Batch visualization of top 6
targets = [
    "layer2.0.conv2:125",  # Channel 125 in layer2.0.conv2 (Change: 874.1881)
    "layer2.2.conv3:297",  # Channel 297 in layer2.2.conv3 (Change: 832.5737)
    "layer3.3.conv2:222",  # Channel 222 in layer3.3.conv2 (Change: 795.9881)
    "layer2.3.conv2:33",  # Channel 33 in layer2.3.conv2 (Change: 709.2872)
    "layer2.0.conv1:0",  # Channel 0 in layer2.0.conv1 (Change: 352.0150)
    "layer2.2.conv3:213",  # Channel 213 in layer2.2.conv3 (Change: 204.4977)
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
