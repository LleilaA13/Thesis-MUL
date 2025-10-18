# Unified Multi-Experiment Lucent Visualization Commands
# Generated from 10%, 20%, 30% forgetting analysis

from lucent.optvis import render
import matplotlib.pyplot as plt

# TOP TARGETS ACROSS ALL EXPERIMENTS
# These are the most influenced components from any experiment

# 1. Channel 902 in layer3.0.conv3 (Change: 3978.5476) (from 10% Random Data Forgetting)
img_all_1 = render.render_vis(model, "layer3.0.conv3:902", show_inline=True, thresholds=(512,))

# 2. Channel 199 in layer2.2.conv3 (Change: 1791.2668) (from 30% Random Data Forgetting)
img_all_2 = render.render_vis(model, "layer2.2.conv3:199", show_inline=True, thresholds=(512,))

# 3. Channel 107 in layer2.0.conv3 (Change: 1710.7089) (from 30% Random Data Forgetting)
img_all_3 = render.render_vis(model, "layer2.0.conv3:107", show_inline=True, thresholds=(512,))

# 4. Channel 387 in layer2.2.conv3 (Change: 1617.4023) (from 20% Random Data Forgetting)
img_all_4 = render.render_vis(model, "layer2.2.conv3:387", show_inline=True, thresholds=(512,))

# 5. Channel 301 in layer3.5.conv3 (Change: 1586.1949) (from 30% Random Data Forgetting)
img_all_5 = render.render_vis(model, "layer3.5.conv3:301", show_inline=True, thresholds=(512,))

# 6. Channel 368 in layer2.3.conv3 (Change: 1251.9911) (from 30% Random Data Forgetting)
img_all_6 = render.render_vis(model, "layer2.3.conv3:368", show_inline=True, thresholds=(512,))

# 7. Channel 793 in layer3.0.conv3 (Change: 1097.4364) (from 10% Random Data Forgetting)
img_all_7 = render.render_vis(model, "layer3.0.conv3:793", show_inline=True, thresholds=(512,))

# 8. Channel 53 in layer1.2.conv3 (Change: 791.7326) (from 20% Random Data Forgetting)
img_all_8 = render.render_vis(model, "layer1.2.conv3:53", show_inline=True, thresholds=(512,))

# 9. Channel 424 in layer4.0.conv3 (Change: 755.0659) (from 20% Random Data Forgetting)
img_all_9 = render.render_vis(model, "layer4.0.conv3:424", show_inline=True, thresholds=(512,))

# 10. Channel 23 in conv1 (Change: 633.7079) (from 10% Random Data Forgetting)
img_all_10 = render.render_vis(model, "conv1:23", show_inline=True, thresholds=(512,))


# 10% RANDOM DATA FORGETTING SPECIFIC TARGETS
# Top targets specifically from 10% Random Data Forgetting

# 10percent_1. Channel 902 in layer3.0.conv3 (Change: 3978.5476)
img_10percent_1 = render.render_vis(model, "layer3.0.conv3:902", show_inline=True, thresholds=(512,))

# 10percent_2. Channel 793 in layer3.0.conv3 (Change: 1097.4364)
img_10percent_2 = render.render_vis(model, "layer3.0.conv3:793", show_inline=True, thresholds=(512,))

# 10percent_3. Channel 23 in conv1 (Change: 633.7079)
img_10percent_3 = render.render_vis(model, "conv1:23", show_inline=True, thresholds=(512,))

# 10percent_4. Channel 98 in layer3.5.conv2 (Change: 613.1884)
img_10percent_4 = render.render_vis(model, "layer3.5.conv2:98", show_inline=True, thresholds=(512,))

# 10percent_5. Channel 123 in layer3.1.conv2 (Change: 416.7875)
img_10percent_5 = render.render_vis(model, "layer3.1.conv2:123", show_inline=True, thresholds=(512,))


# 20% RANDOM DATA FORGETTING SPECIFIC TARGETS
# Top targets specifically from 20% Random Data Forgetting

# 20percent_1. Channel 387 in layer2.2.conv3 (Change: 1617.4023)
img_20percent_1 = render.render_vis(model, "layer2.2.conv3:387", show_inline=True, thresholds=(512,))

# 20percent_2. Channel 53 in layer1.2.conv3 (Change: 791.7326)
img_20percent_2 = render.render_vis(model, "layer1.2.conv3:53", show_inline=True, thresholds=(512,))

# 20percent_3. Channel 424 in layer4.0.conv3 (Change: 755.0659)
img_20percent_3 = render.render_vis(model, "layer4.0.conv3:424", show_inline=True, thresholds=(512,))

# 20percent_4. Channel 482 in layer3.5.conv3 (Change: 586.0610)
img_20percent_4 = render.render_vis(model, "layer3.5.conv3:482", show_inline=True, thresholds=(512,))

# 20percent_5. Channel 41 in layer1.2.conv1 (Change: 460.2436)
img_20percent_5 = render.render_vis(model, "layer1.2.conv1:41", show_inline=True, thresholds=(512,))


# 30% RANDOM DATA FORGETTING SPECIFIC TARGETS
# Top targets specifically from 30% Random Data Forgetting

# 30percent_1. Channel 199 in layer2.2.conv3 (Change: 1791.2668)
img_30percent_1 = render.render_vis(model, "layer2.2.conv3:199", show_inline=True, thresholds=(512,))

# 30percent_2. Channel 107 in layer2.0.conv3 (Change: 1710.7089)
img_30percent_2 = render.render_vis(model, "layer2.0.conv3:107", show_inline=True, thresholds=(512,))

# 30percent_3. Channel 301 in layer3.5.conv3 (Change: 1586.1949)
img_30percent_3 = render.render_vis(model, "layer3.5.conv3:301", show_inline=True, thresholds=(512,))

# 30percent_4. Channel 368 in layer2.3.conv3 (Change: 1251.9911)
img_30percent_4 = render.render_vis(model, "layer2.3.conv3:368", show_inline=True, thresholds=(512,))

# 30percent_5. Channel 95 in layer2.3.conv2 (Change: 362.0521)
img_30percent_5 = render.render_vis(model, "layer2.3.conv2:95", show_inline=True, thresholds=(512,))


# BATCH COMPARISON VISUALIZATION
# Compare top targets from each experiment side by side

def compare_experiments():
    experiments = {
        "10% Random Data Forgetting": ['layer3.0.conv3:902', 'layer3.0.conv3:793', 'conv1:23'],
        "20% Random Data Forgetting": ['layer2.2.conv3:387', 'layer1.2.conv3:53', 'layer4.0.conv3:424'],
        "30% Random Data Forgetting": ['layer2.2.conv3:199', 'layer2.0.conv3:107', 'layer3.5.conv3:301'],
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
