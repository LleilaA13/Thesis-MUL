# Real Influence Analysis - Lucent Visualization Commands
# Copy these into Google Colab

from lucent.optvis import render
import matplotlib.pyplot as plt

# Top individual targets
# 1. Channel 902 in layer3.0.conv3 (Change: 3978.5476)
img_1 = render.render_vis(model, "layer3.0.conv3:902", show_inline=True, thresholds=(512,))

# 2. Channel 793 in layer3.0.conv3 (Change: 1097.4364)
img_2 = render.render_vis(model, "layer3.0.conv3:793", show_inline=True, thresholds=(512,))

# 3. Channel 23 in conv1 (Change: 633.7079)
img_3 = render.render_vis(model, "conv1:23", show_inline=True, thresholds=(512,))

# 4. Channel 98 in layer3.5.conv2 (Change: 613.1884)
img_4 = render.render_vis(model, "layer3.5.conv2:98", show_inline=True, thresholds=(512,))

# 5. Channel 123 in layer3.1.conv2 (Change: 416.7875)
img_5 = render.render_vis(model, "layer3.1.conv2:123", show_inline=True, thresholds=(512,))

# 6. Channel 200 in layer3.4.conv2 (Change: 262.2291)
img_6 = render.render_vis(model, "layer3.4.conv2:200", show_inline=True, thresholds=(512,))

# 7. Channel 651 in layer3.0.conv3 (Change: 241.5894)
img_7 = render.render_vis(model, "layer3.0.conv3:651", show_inline=True, thresholds=(512,))

# 8. Channel 390 in layer4.0.conv2 (Change: 186.5821)
img_8 = render.render_vis(model, "layer4.0.conv2:390", show_inline=True, thresholds=(512,))

# 9. Channel 35 in conv1 (Change: 161.6789)
img_9 = render.render_vis(model, "conv1:35", show_inline=True, thresholds=(512,))

# 10. Channel 175 in layer3.0.conv2 (Change: 154.6189)
img_10 = render.render_vis(model, "layer3.0.conv2:175", show_inline=True, thresholds=(512,))


# Batch visualization of top 6
targets = [
    "layer3.0.conv3:902",  # Channel 902 in layer3.0.conv3 (Change: 3978.5476)
    "layer3.0.conv3:793",  # Channel 793 in layer3.0.conv3 (Change: 1097.4364)
    "conv1:23",  # Channel 23 in conv1 (Change: 633.7079)
    "layer3.5.conv2:98",  # Channel 98 in layer3.5.conv2 (Change: 613.1884)
    "layer3.1.conv2:123",  # Channel 123 in layer3.1.conv2 (Change: 416.7875)
    "layer3.4.conv2:200",  # Channel 200 in layer3.4.conv2 (Change: 262.2291)
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
