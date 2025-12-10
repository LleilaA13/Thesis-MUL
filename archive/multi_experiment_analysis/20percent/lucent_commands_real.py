# Real Influence Analysis - Lucent Visualization Commands
# Copy these into Google Colab

from lucent.optvis import render
import matplotlib.pyplot as plt

# Top individual targets
# 1. Channel 52 in layer1.2.conv2 (Change: 2401.4683)
img_1 = render.render_vis(model, "layer1.2.conv2:52", show_inline=True, thresholds=(512,))

# 2. Channel 793 in layer3.1.conv3 (Change: 1375.1271)
img_2 = render.render_vis(model, "layer3.1.conv3:793", show_inline=True, thresholds=(512,))

# 3. Channel 401 in layer3.1.conv3 (Change: 1360.5739)
img_3 = render.render_vis(model, "layer3.1.conv3:401", show_inline=True, thresholds=(512,))

# 4. Channel 807 in layer4.0.conv3 (Change: 622.3548)
img_4 = render.render_vis(model, "layer4.0.conv3:807", show_inline=True, thresholds=(512,))

# 5. Channel 145 in layer3.0.conv2 (Change: 559.0394)
img_5 = render.render_vis(model, "layer3.0.conv2:145", show_inline=True, thresholds=(512,))

# 6. Channel 561 in layer4.0.conv3 (Change: 390.4918)
img_6 = render.render_vis(model, "layer4.0.conv3:561", show_inline=True, thresholds=(512,))

# 7. Channel 1963 in layer4.0.conv3 (Change: 372.0911)
img_7 = render.render_vis(model, "layer4.0.conv3:1963", show_inline=True, thresholds=(512,))

# 8. Channel 33 in conv1 (Change: 283.1871)
img_8 = render.render_vis(model, "conv1:33", show_inline=True, thresholds=(512,))

# 9. Channel 1251 in layer4.0.conv3 (Change: 248.1846)
img_9 = render.render_vis(model, "layer4.0.conv3:1251", show_inline=True, thresholds=(512,))

# 10. Channel 8 in conv1 (Change: 218.8287)
img_10 = render.render_vis(model, "conv1:8", show_inline=True, thresholds=(512,))


# Batch visualization of top 6
targets = [
    "layer1.2.conv2:52",  # Channel 52 in layer1.2.conv2 (Change: 2401.4683)
    "layer3.1.conv3:793",  # Channel 793 in layer3.1.conv3 (Change: 1375.1271)
    "layer3.1.conv3:401",  # Channel 401 in layer3.1.conv3 (Change: 1360.5739)
    "layer4.0.conv3:807",  # Channel 807 in layer4.0.conv3 (Change: 622.3548)
    "layer3.0.conv2:145",  # Channel 145 in layer3.0.conv2 (Change: 559.0394)
    "layer4.0.conv3:561",  # Channel 561 in layer4.0.conv3 (Change: 390.4918)
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
