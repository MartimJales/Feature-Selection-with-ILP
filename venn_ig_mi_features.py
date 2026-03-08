import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib_venn import venn2

matplotlib.rcParams['pdf.fonttype'] = 42  # Type 42 = TrueType
matplotlib.rcParams['ps.fonttype'] = 42

plt.rcParams['font.family'] = 'Liberation Sans'

# Load feature rankings
df = pd.read_csv("reports/feature_analysis/feature_rankings_top30.csv")

# Define different top-K values to analyze
top_k_values = [10, 20, 30]

# Colors
color_ig = '#BAE1FF'
color_mi = '#FF5733'
color_both = "#4EC8DD"  # Purple for intersection
color_edge = '#000000'
width_edge = 1.5
fontsize_title = 14

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, top_k in enumerate(top_k_values):
    # Get top-K features by Information Gain
    df_sorted_ig = df.nlargest(top_k, 'information_gain')
    features_ig = set(df_sorted_ig['feature'])

    # Get top-K features by Mutual Information
    df_sorted_mi = df.nlargest(top_k, 'mutual_information')
    features_mi = set(df_sorted_mi['feature'])

    # Create Venn diagram
    plt.subplot(1, 3, idx + 1)
    venn = venn2(
        [features_ig, features_mi],
        set_labels=('', ''),
        set_colors=(color_ig, color_mi),
        ax=axes[idx]
    )

    # Change intersection color
    if venn.get_patch_by_id('11'):
        venn.get_patch_by_id('11').set_facecolor(color_both)

    # Add borders
    for subset in ('10', '01', '11'):
        if venn.get_patch_by_id(subset):
            venn.get_patch_by_id(subset).set_edgecolor(color_edge)
            venn.get_patch_by_id(subset).set_linewidth(width_edge)

    # Set font size for labels
    for text in venn.subset_labels:
        if text:
            text.set_fontsize(12)

    plt.title(f"Top-{top_k} Features", fontsize=fontsize_title)

    # Calculate overlap percentage
    overlap = len(features_ig & features_mi)
    overlap_pct = (overlap / top_k) * 100
    print(f"\nTop-{top_k}:")
    print(f"  IG only: {len(features_ig - features_mi)}")
    print(f"  MI only: {len(features_mi - features_ig)}")
    print(f"  Both: {overlap} ({overlap_pct:.1f}%)")

    # Print features that differ
    if features_ig != features_mi:
        print(f"  IG unique: {features_ig - features_mi}")
        print(f"  MI unique: {features_mi - features_ig}")

# Add legend
custom_handles = [
    Patch(facecolor=color_ig, edgecolor='black', linewidth=1, alpha=0.7),
    Patch(facecolor=color_mi, edgecolor='black', linewidth=1, alpha=0.7),
    Patch(facecolor=color_both, edgecolor='black', linewidth=1, alpha=0.7)
]
fig.legend(
    custom_handles,
    ['Information Gain', 'Mutual Information', 'Both (Intersection)'],
    loc='lower center',
    ncol=3,
    fontsize=14
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("reports/feature_analysis/venn_ig_mi_comparison.png", dpi=300, bbox_inches='tight')
print("\n✓ Venn diagrams saved to reports/feature_analysis/venn_ig_mi_comparison.png")
