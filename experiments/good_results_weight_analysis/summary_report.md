# Random Data Forgetting - Weight Analysis Summary
Generated on: 2025-10-13 14:22:34


## Experiment: random_forgetting_30percent_RL_tweak_conservative
- **Forget Ratio**: forgetting
- **Mask Threshold**: RL
- **Unlearn Method**: conservative
- **Most Affected Layer**: layer1.1.bn1.running_mean (22.350573)
- **Least Affected Layer**: normalize.mean (0.000000)
- **Top 3 Most Changed Weights**:
  - layer3.5.conv1.weight (change: 687802.812500)
  - layer4.0.conv2.weight (change: 598992.562500)
  - layer3.0.conv2.weight (change: 516677.875000)


## Experiment: random_forgetting_10percent_RL_conservative
- **Most Affected Layer**: layer1.1.bn1.running_mean (23.775593)
- **Least Affected Layer**: normalize.mean (0.000000)
- **Top 3 Most Changed Weights**:
  - layer2.0.conv2.weight (change: 577243.750000)
  - layer3.1.conv3.weight (change: 433590.031250)
  - layer3.5.conv1.weight (change: 414986.062500)


## Experiment: random_forgetting_20percent_RL_tweak_conservative
- **Forget Ratio**: forgetting
- **Mask Threshold**: RL
- **Unlearn Method**: conservative
- **Most Affected Layer**: bn1.running_var (17.206081)
- **Least Affected Layer**: normalize.mean (0.000000)
- **Top 3 Most Changed Weights**:
  - layer4.1.conv2.weight (change: 494795.875000)
  - layer3.5.conv1.weight (change: 457472.343750)
  - layer4.0.conv3.weight (change: 418490.843750)

