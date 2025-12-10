# Real Influence Analysis - Lucent Visualization Commands
# Copy these into Google Colab

from lucent.optvis import render
import matplotlib.pyplot as plt

# Top individual targets
# 1. Channel 182 in layer3.2.conv3 (Change: 4908.4883)
img_1 = render.render_vis(model, "layer3.2.conv3:182", show_inline=True, thresholds=(512,))

# 2. Channel 106 in layer4.1.conv3 (Change: 1635.2512)
img_2 = render.render_vis(model, "layer4.1.conv3:106", show_inline=True, thresholds=(512,))

# 3. Channel 549 in layer4.0.conv3 (Change: 1207.2595)
img_3 = render.render_vis(model, "layer4.0.conv3:549", show_inline=True, thresholds=(512,))

# 4. Channel 1787 in layer4.0.conv3 (Change: 796.3290)
img_4 = render.render_vis(model, "layer4.0.conv3:1787", show_inline=True, thresholds=(512,))

# 5. Channel 430 in layer4.1.conv3 (Change: 603.2018)
img_5 = render.render_vis(model, "layer4.1.conv3:430", show_inline=True, thresholds=(512,))

# 6. Channel 351 in layer3.2.conv3 (Change: 529.8499)
img_6 = render.render_vis(model, "layer3.2.conv3:351", show_inline=True, thresholds=(512,))

# 7. Channel 1743 in layer4.0.conv3 (Change: 456.9725)
img_7 = render.render_vis(model, "layer4.0.conv3:1743", show_inline=True, thresholds=(512,))

# 8. Channel 65 in layer4.0.conv2 (Change: 380.8553)
img_8 = render.render_vis(model, "layer4.0.conv2:65", show_inline=True, thresholds=(512,))

# 9. Channel 1600 in layer4.0.conv3 (Change: 364.5104)
img_9 = render.render_vis(model, "layer4.0.conv3:1600", show_inline=True, thresholds=(512,))

# 10. Channel 399 in layer4.1.conv3 (Change: 333.8979)
img_10 = render.render_vis(model, "layer4.1.conv3:399", show_inline=True, thresholds=(512,))


# Batch visualization of top 6
targets = [
    "layer3.2.conv3:182",  # Channel 182 in layer3.2.conv3 (Change: 4908.4883)
    "layer4.1.conv3:106",  # Channel 106 in layer4.1.conv3 (Change: 1635.2512)
    "layer4.0.conv3:549",  # Channel 549 in layer4.0.conv3 (Change: 1207.2595)
    "layer4.0.conv3:1787",  # Channel 1787 in layer4.0.conv3 (Change: 796.3290)
    "layer4.1.conv3:430",  # Channel 430 in layer4.1.conv3 (Change: 603.2018)
    "layer3.2.conv3:351",  # Channel 351 in layer3.2.conv3 (Change: 529.8499)
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
