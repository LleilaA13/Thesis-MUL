# Unified Multi-Experiment Lucent Visualization Commands
# Generated from 10%, 20%, 30% forgetting analysis

from lucent.optvis import render
import matplotlib.pyplot as plt

# TOP TARGETS ACROSS ALL EXPERIMENTS
# These are the most influenced components from any experiment

# 1. Channel 274 in layer3.1.conv3 (Change: 3675.7275) (from 30% Random Data Forgetting)
img_all_1 = render.render_vis(model, "layer3.1.conv3:274", show_inline=True, thresholds=(512,))

# 2. Channel 52 in layer1.2.conv2 (Change: 2401.4683) (from 20% Random Data Forgetting)
img_all_2 = render.render_vis(model, "layer1.2.conv2:52", show_inline=True, thresholds=(512,))

# 3. Channel 51 in layer3.0.conv1 (Change: 1786.9342) (from 30% Random Data Forgetting)
img_all_3 = render.render_vis(model, "layer3.0.conv1:51", show_inline=True, thresholds=(512,))

# 4. Channel 793 in layer3.1.conv3 (Change: 1375.1271) (from 20% Random Data Forgetting)
img_all_4 = render.render_vis(model, "layer3.1.conv3:793", show_inline=True, thresholds=(512,))

# 5. Channel 401 in layer3.1.conv3 (Change: 1360.5739) (from 20% Random Data Forgetting)
img_all_5 = render.render_vis(model, "layer3.1.conv3:401", show_inline=True, thresholds=(512,))

# 6. Channel 44 in layer2.3.conv2 (Change: 1058.8596) (from 30% Random Data Forgetting)
img_all_6 = render.render_vis(model, "layer2.3.conv2:44", show_inline=True, thresholds=(512,))

# 7. Channel 31 in layer1.0.conv3 (Change: 680.7674) (from 30% Random Data Forgetting)
img_all_7 = render.render_vis(model, "layer1.0.conv3:31", show_inline=True, thresholds=(512,))

# 8. Channel 807 in layer4.0.conv3 (Change: 622.3548) (from 20% Random Data Forgetting)
img_all_8 = render.render_vis(model, "layer4.0.conv3:807", show_inline=True, thresholds=(512,))

# 9. Channel 145 in layer3.0.conv2 (Change: 559.0394) (from 20% Random Data Forgetting)
img_all_9 = render.render_vis(model, "layer3.0.conv2:145", show_inline=True, thresholds=(512,))

# 10. Channel 30 in layer3.0.conv1 (Change: 501.8506) (from 10% Random Data Forgetting)
img_all_10 = render.render_vis(model, "layer3.0.conv1:30", show_inline=True, thresholds=(512,))


# 10% RANDOM DATA FORGETTING SPECIFIC TARGETS
# Top targets specifically from 10% Random Data Forgetting

# 10percent_1. Channel 30 in layer3.0.conv1 (Change: 501.8506)
img_10percent_1 = render.render_vis(model, "layer3.0.conv1:30", show_inline=True, thresholds=(512,))

# 10percent_2. Channel 339 in layer4.0.conv3 (Change: 388.5698)
img_10percent_2 = render.render_vis(model, "layer4.0.conv3:339", show_inline=True, thresholds=(512,))

# 10percent_3. Channel 1686 in layer4.0.conv3 (Change: 387.4626)
img_10percent_3 = render.render_vis(model, "layer4.0.conv3:1686", show_inline=True, thresholds=(512,))

# 10percent_4. Channel 250 in layer4.1.conv2 (Change: 176.8248)
img_10percent_4 = render.render_vis(model, "layer4.1.conv2:250", show_inline=True, thresholds=(512,))

# 10percent_5. Channel 211 in layer4.0.conv3 (Change: 167.6543)
img_10percent_5 = render.render_vis(model, "layer4.0.conv3:211", show_inline=True, thresholds=(512,))


# 20% RANDOM DATA FORGETTING SPECIFIC TARGETS
# Top targets specifically from 20% Random Data Forgetting

# 20percent_1. Channel 52 in layer1.2.conv2 (Change: 2401.4683)
img_20percent_1 = render.render_vis(model, "layer1.2.conv2:52", show_inline=True, thresholds=(512,))

# 20percent_2. Channel 793 in layer3.1.conv3 (Change: 1375.1271)
img_20percent_2 = render.render_vis(model, "layer3.1.conv3:793", show_inline=True, thresholds=(512,))

# 20percent_3. Channel 401 in layer3.1.conv3 (Change: 1360.5739)
img_20percent_3 = render.render_vis(model, "layer3.1.conv3:401", show_inline=True, thresholds=(512,))

# 20percent_4. Channel 807 in layer4.0.conv3 (Change: 622.3548)
img_20percent_4 = render.render_vis(model, "layer4.0.conv3:807", show_inline=True, thresholds=(512,))

# 20percent_5. Channel 145 in layer3.0.conv2 (Change: 559.0394)
img_20percent_5 = render.render_vis(model, "layer3.0.conv2:145", show_inline=True, thresholds=(512,))


# 30% RANDOM DATA FORGETTING SPECIFIC TARGETS
# Top targets specifically from 30% Random Data Forgetting

# 30percent_1. Channel 274 in layer3.1.conv3 (Change: 3675.7275)
img_30percent_1 = render.render_vis(model, "layer3.1.conv3:274", show_inline=True, thresholds=(512,))

# 30percent_2. Channel 51 in layer3.0.conv1 (Change: 1786.9342)
img_30percent_2 = render.render_vis(model, "layer3.0.conv1:51", show_inline=True, thresholds=(512,))

# 30percent_3. Channel 44 in layer2.3.conv2 (Change: 1058.8596)
img_30percent_3 = render.render_vis(model, "layer2.3.conv2:44", show_inline=True, thresholds=(512,))

# 30percent_4. Channel 31 in layer1.0.conv3 (Change: 680.7674)
img_30percent_4 = render.render_vis(model, "layer1.0.conv3:31", show_inline=True, thresholds=(512,))

# 30percent_5. Channel 365 in layer4.1.conv2 (Change: 247.3791)
img_30percent_5 = render.render_vis(model, "layer4.1.conv2:365", show_inline=True, thresholds=(512,))


# BATCH COMPARISON VISUALIZATION
# Compare top targets from each experiment side by side

def compare_experiments():
    experiments = {
        "10% Random Data Forgetting": ['layer3.0.conv1:30', 'layer4.0.conv3:339', 'layer4.0.conv3:1686'],
        "20% Random Data Forgetting": ['layer1.2.conv2:52', 'layer3.1.conv3:793', 'layer3.1.conv3:401'],
        "30% Random Data Forgetting": ['layer3.1.conv3:274', 'layer3.0.conv1:51', 'layer2.3.conv2:44'],
    }
    
    fig, axes = plt.subplots(len(experiments), 3, figsize=(15, 5*len(experiments)))
    if len(experiments) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (exp_name, targets) in enumerate(experiments.items()):
        for j, target in enumerate(targets):
            img = render.render_vis(model, target, show_inline=False, thresholds=(256,))
            if hasattr(img, 'cpu'):
                img_np = img.cpu().numpy().transpose(1, 2, 0)
            else:
                img_np = np.array(img)
            axes[i, j].imshow(img_np)
            axes[i, j].set_title(f"{exp_name}\n{target}", fontsize=10)
            axes[i, j].axis('off')
    
    plt.suptitle('Multi-Experiment Comparison: Most Influenced Components', fontsize=16)
    plt.tight_layout()
    plt.show()

# Run the comparison
compare_experiments()
