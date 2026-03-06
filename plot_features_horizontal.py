import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load data
df = pd.read_csv("reports/feature_analysis/feature_rankings_top30.csv")
df.columns = df.columns.str.strip()

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

# Title
title = "Top-30 Features by Information Gain & Mutual Information"
fig.text(0.5, 0.95, title, ha='center', fontsize=18, fontweight='bold')

# Features list
features = df['feature'].tolist()

# Create text with features
y_position = 0.90
fontsize = 11

for i, feature in enumerate(features, 1):
    # Format: "1. feature_name"
    text = f"{i}. {feature}"
    fig.text(0.1, y_position, text, fontsize=fontsize, family='monospace', verticalalignment='top')
    y_position -= 0.029

# Save
plt.savefig("reports/feature_analysis/features_list.png", dpi=300, bbox_inches='tight', facecolor='white')
# plt.savefig("reports/feature_analysis/features_list.pdf", bbox_inches='tight', facecolor='white')

print("✓ Lista de features salva em reports/feature_analysis/features_list.png")
