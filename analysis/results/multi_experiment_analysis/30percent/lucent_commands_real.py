# Real Influence Analysis - Lucent Visualization Commands
# Copy these into Google Colab

from lucent.optvis import render
import matplotlib.pyplot as plt

# Top individual targets
# 1. Channel 199 in layer2.2.conv3 (Change: 1791.2668)
img_1 = render.render_vis(model, "layer2.2.conv3:199", show_inline=True, thresholds=(512,))

# 2. Channel 107 in layer2.0.conv3 (Change: 1710.7089)
img_2 = render.render_vis(model, "layer2.0.conv3:107", show_inline=True, thresholds=(512,))

# 3. Channel 301 in layer3.5.conv3 (Change: 1586.1949)
img_3 = render.render_vis(model, "layer3.5.conv3:301", show_inline=True, thresholds=(512,))

# 4. Channel 368 in layer2.3.conv3 (Change: 1251.9911)
img_4 = render.render_vis(model, "layer2.3.conv3:368", show_inline=True, thresholds=(512,))

# 5. Channel 95 in layer2.3.conv2 (Change: 362.0521)
img_5 = render.render_vis(model, "layer2.3.conv2:95", show_inline=True, thresholds=(512,))

# 6. Channel 165 in layer2.3.conv3 (Change: 260.4688)
img_6 = render.render_vis(model, "layer2.3.conv3:165", show_inline=True, thresholds=(512,))

# 7. Channel 133 in layer3.0.conv2 (Change: 246.0330)
img_7 = render.render_vis(model, "layer3.0.conv2:133", show_inline=True, thresholds=(512,))

# 8. Channel 115 in layer3.4.conv2 (Change: 223.8363)
img_8 = render.render_vis(model, "layer3.4.conv2:115", show_inline=True, thresholds=(512,))

# 9. Channel 254 in layer4.1.conv2 (Change: 215.2360)
img_9 = render.render_vis(model, "layer4.1.conv2:254", show_inline=True, thresholds=(512,))

# 10. Channel 48 in layer3.0.conv2 (Change: 203.0474)
img_10 = render.render_vis(model, "layer3.0.conv2:48", show_inline=True, thresholds=(512,))


# Batch visualization of top 6
targets = [
    "layer2.2.conv3:199",  # Channel 199 in layer2.2.conv3 (Change: 1791.2668)
    "layer2.0.conv3:107",  # Channel 107 in layer2.0.conv3 (Change: 1710.7089)
    "layer3.5.conv3:301",  # Channel 301 in layer3.5.conv3 (Change: 1586.1949)
    "layer2.3.conv3:368",  # Channel 368 in layer2.3.conv3 (Change: 1251.9911)
    "layer2.3.conv2:95",  # Channel 95 in layer2.3.conv2 (Change: 362.0521)
    "layer2.3.conv3:165",  # Channel 165 in layer2.3.conv3 (Change: 260.4688)
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
