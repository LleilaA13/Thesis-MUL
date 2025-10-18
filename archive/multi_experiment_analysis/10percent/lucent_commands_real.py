# Real Influence Analysis - Lucent Visualization Commands
# Copy these into Google Colab

from lucent.optvis import render
import matplotlib.pyplot as plt

# Top individual targets
# 1. Channel 30 in layer3.0.conv1 (Change: 501.8506)
img_1 = render.render_vis(model, "layer3.0.conv1:30", show_inline=True, thresholds=(512,))

# 2. Channel 339 in layer4.0.conv3 (Change: 388.5698)
img_2 = render.render_vis(model, "layer4.0.conv3:339", show_inline=True, thresholds=(512,))

# 3. Channel 1686 in layer4.0.conv3 (Change: 387.4626)
img_3 = render.render_vis(model, "layer4.0.conv3:1686", show_inline=True, thresholds=(512,))

# 4. Channel 250 in layer4.1.conv2 (Change: 176.8248)
img_4 = render.render_vis(model, "layer4.1.conv2:250", show_inline=True, thresholds=(512,))

# 5. Channel 211 in layer4.0.conv3 (Change: 167.6543)
img_5 = render.render_vis(model, "layer4.0.conv3:211", show_inline=True, thresholds=(512,))

# 6. Channel 137 in layer4.0.conv3 (Change: 157.4682)
img_6 = render.render_vis(model, "layer4.0.conv3:137", show_inline=True, thresholds=(512,))

# 7. Channel 7 in conv1 (Change: 141.4556)
img_7 = render.render_vis(model, "conv1:7", show_inline=True, thresholds=(512,))

# 8. Channel 18 in conv1 (Change: 138.7276)
img_8 = render.render_vis(model, "conv1:18", show_inline=True, thresholds=(512,))

# 9. Channel 243 in layer4.0.conv3 (Change: 138.0210)
img_9 = render.render_vis(model, "layer4.0.conv3:243", show_inline=True, thresholds=(512,))

# 10. Channel 801 in layer4.0.conv3 (Change: 134.9254)
img_10 = render.render_vis(model, "layer4.0.conv3:801", show_inline=True, thresholds=(512,))


# Batch visualization of top 6
targets = [
    "layer3.0.conv1:30",  # Channel 30 in layer3.0.conv1 (Change: 501.8506)
    "layer4.0.conv3:339",  # Channel 339 in layer4.0.conv3 (Change: 388.5698)
    "layer4.0.conv3:1686",  # Channel 1686 in layer4.0.conv3 (Change: 387.4626)
    "layer4.1.conv2:250",  # Channel 250 in layer4.1.conv2 (Change: 176.8248)
    "layer4.0.conv3:211",  # Channel 211 in layer4.0.conv3 (Change: 167.6543)
    "layer4.0.conv3:137",  # Channel 137 in layer4.0.conv3 (Change: 157.4682)
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
