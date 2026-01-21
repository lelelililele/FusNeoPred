import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------
# 1. Prepare data
# ---------------------------------------------------------
filename = 'T5_score_RawPre'

try:
    df = pd.read_csv(filename, sep='\s+', header=None, names=['ID', 'Rank1', 'Rank2'])
except FileNotFoundError:
    print(f"File {filename} not found, generating simulated test data...")
    # Generate test data: simulate some data falling out of top 100
    np.random.seed(42)
    rank1 = np.arange(1, 6001)
    # Rank2 fluctuates around Rank1, some become very large (>100)
    rank2 = rank1 + np.random.randint(-20, 50, size=6000) 
    rank2 = np.abs(rank2)
    rank2[rank2 == 0] = 1
    
    df = pd.DataFrame({
        'ID': [f'Sample_{i}' for i in range(1, 6001)],
        'Rank1': rank1,
        'Rank2': rank2
    })

# ---------------------------------------------------------
# 2. Data processing and statistics
# ---------------------------------------------------------

# Step 1: Only consider top 100 in Rank1
df_source = df[df['Rank1'] <= 100].copy()
total_source = len(df_source)  # Should be 100

# Step 2: Filter data that "remains in top 100" (Rank2 <= 100)
df_kept = df_source[df_source['Rank2'] <= 100].copy()

# Step 3: Statistical results
count_kept = len(df_kept)
count_lost = total_source - count_kept
retention_rate = (count_kept / total_source) * 100

print(f"Statistical results:")
print(f"Among the top 100 in Rank1, {count_kept} remain in the top 100 in Rank2.")
print(f"{count_lost} dropped out of the top 100.")

# ---------------------------------------------------------
# 3. Grouping and color settings
# ---------------------------------------------------------

# Set grouping: 10 per group
group_size = 10

# To ensure color mapping is based on original 1-100 (10 groups total), fix normalization range
# Group 0 (1-10) -> Group 9 (91-100)
cmap = plt.get_cmap('viridis')
norm = mcolors.Normalize(vmin=0, vmax=9) 

# Generate colors for retained data
df_kept['Group'] = (df_kept['Rank1'] - 1) // group_size
colors_kept = cmap(norm(df_kept['Group'].values))

# Generate colors for source data (left points) to also plot dropped points on left (optional, but recommended to show gaps)
df_source['Group'] = (df_source['Rank1'] - 1) // group_size
colors_source = cmap(norm(df_source['Group'].values))

# ---------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 14))

# --- A. Draw lines (only for retained data) ---
segments = []
for r1, r2 in zip(df_kept['Rank1'], df_kept['Rank2']):
    segments.append([(0, r1), (1, r2)])

lc = LineCollection(segments, colors=colors_kept, linewidths=1.5, alpha=0.8)
ax.add_collection(lc)

# --- B. Draw endpoints ---

# 1. Left points: I recommend plotting all Rank1 top 100 points.
#    This way, if a point has no connecting line, you know it dropped out (hollow or unconnected).
#    Plot all Rank1 points here
ax.scatter([0]*len(df_source), df_source['Rank1'], c=colors_source, s=40, marker='o', 
           edgecolors='black', linewidth=0.5, zorder=3, label='Rank 1 (Baseline)')

# 2. Right points: Only plot retained points (Rank2 <= 100)
ax.scatter([1]*len(df_kept), df_kept['Rank2'], c=colors_kept, s=40, marker='s', 
           edgecolors='black', linewidth=0.5, zorder=3, label='Rank 2 (Retained)')

# ---------------------------------------------------------
# 5. Annotations and statistical text
# ---------------------------------------------------------

# Annotate left side: start of each group (1, 11, 21...)
for i, row in df_source.iterrows():
    r1 = int(row['Rank1'])
    if (r1 - 1) % group_size == 0:
        ax.text(-0.02, r1, str(r1), ha='right', va='center', fontsize=10, fontweight='bold', color='black')

# Annotate right side: Only annotate retained data, and only label group leaders (for clarity)
# Alternatively annotate all retained points, here choosing to annotate "key retained points"
for i, row in df_kept.iterrows():
    r1 = int(row['Rank1'])
    r2 = int(row['Rank2'])
    
    # Strategy: Annotate if this is the starting point of its group (10 per group) and it's still in the plot
    if (r1 - 1) % group_size == 0:
        ax.text(1.02, r2, str(r2), ha='left', va='center', fontsize=10, fontweight='bold', color='black')

# Add prominent statistics box at top of chart
stats_text = (f"Statistics (Top 100 Analysis):\n"
              f"----------------------------\n"
              f"Rank 1 Total : 100\n"
              f"Rank 2 <=100 : {count_kept} (Retained)\n"
              f"Rank 2 > 100 : {count_lost} (Dropped)")

# Add text box (coordinates are relative to canvas, 0.05, 0.95 is top-left)
props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=props, zorder=5)

# ---------------------------------------------------------
# 6. Chart beautification
# ---------------------------------------------------------

ax.set_xlim(-0.15, 1.15)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Rank 1 (Top 100)', 'Rank 2 (Filtered <= 100)'], fontsize=12, fontweight='bold')

# Y-axis only shows 0 to 100 (with a little extra margin)
ax.set_ylim(0, 105)

# Remove borders
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Grid lines
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.title('Retention of Top 100 Items\n(Rank 1-10 Red -> 91-100 Purple)', fontsize=14, pad=20)
plt.tight_layout()

# Save or display
plt.savefig('T5top100_retention_stats.pdf', dpi=300)
# plt.show()