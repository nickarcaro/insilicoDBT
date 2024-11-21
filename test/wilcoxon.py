import pandas as pd
from scipy.stats import wilcoxon

# Data from the table (F1-score values for each model)
models = [
    "U-NET",
    "FCN resnet50",
    "FCN resnet101",
    "FCN50 Pretrained",
    "FCN101 Pretrained",
    "DeepLabv3 resnet50",
    "DeepLabv3 resnet101",
    "DeepLabv3 resnet50 Pretrained",
    "DeepLabv3 resnet101 Pretrained",
    "DeepLabv3+ resnet50",
    "DeepLabv3+ resnet101",
    "DeepLabv3+ resnet50 Pretrained",
    "DeepLabv3+ resnet101 Pretrained",
]

f1_scores = [
    92.22,
    90.73,
    90.80,
    86.96,
    88.71,
    90.95,
    90.76,
    88.84,
    89.77,
    84.86,
    83.08,
    89.04,
    89.40,
]

# Split into UNet vs other models for paired comparison
unet_score = [92.22] * len(f1_scores[1:])  # UNet score repeated for comparison
other_scores = f1_scores[1:]  # Scores of other models

# Perform Wilcoxon signed-rank test
stat, p_value = wilcoxon(unet_score, other_scores)

# Compile the results
p_value, stat
