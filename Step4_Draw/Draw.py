import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# Read data
# Assuming file is tab-separated
df = pd.read_csv('T5_score_RawPre', sep='\t', header=None, names=['ID', 'sort1', 'sort2'])

# Sort by first sort order
df = df.sort_values('sort1').reset_index(drop=True)

# Create color groups
df['color_group'] = (df['sort1'] - 1) // 100  # 0-99 as group 0, 100-199 as group 1, and so on
num_groups = df['color_group'].nunique()

# Create color map
colors = cm.viridis(np.linspace(0, 1, num_groups))

# Create figure
plt.figure(figsize=(15, 8))

# Plot points for each group
for i, group_num in enumerate(sorted(df['color_group'].unique())):
    group_data = df[df['color_group'] == group_num]
    color = colors[i]
    
    # Plot points
    plt.scatter(group_data['sort1'], group_data['sort2'], 
                color=color, s=20, alpha=0.7,
                label=f'Group {group_num*100+1}-{(group_num+1)*100}')
    
    # Draw connecting lines
    # Need to sort each group internally by first sort order
    group_data_sorted = group_data.sort_values('sort1')
#    plt.plot(group_data_sorted['sort1'], group_data_sorted['sort2'], 
#             color=color, linewidth=0.5, alpha=0.5)

# Add overall connecting lines
df_sorted = df.sort_values('sort1')
#plt.plot(df_sorted['sort1'], df_sorted['sort2'], 
#         color='gray', linewidth=0.3, alpha=0.3, zorder=1)

# Set figure properties
plt.xlabel('First Sort Order', fontsize=12)
plt.ylabel('Second Sort Order', fontsize=12)
plt.title('Connection Plot: First Sort vs Second Sort (Colored by Groups of 100)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Add legend (if too many groups, can omit or simplify legend)
if num_groups <= 20:
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.subplots_adjust(right=0.8)
else:
    # When too many groups, only show legend for first few groups
    print(f"Too many groups ({num_groups}), legend omitted. Using colorbar instead.")
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, 
                               norm=plt.Normalize(vmin=df['color_group'].min()*100, 
                                                   vmax=(df['color_group'].max()+1)*100))
    sm.set_array([])
    cbar = plt.colorbar(sm, pad=0.01)
    cbar.set_label('First Sort Range (Start Value)', fontsize=10)


# Save figure
plt.savefig('T5sort_connection_plot.pdf', dpi=300, bbox_inches='tight')
print("Plot saved as 'T5sort_connection_plot.pdf'")
plt.show()

# Display basic data information
print(f"Total rows: {len(df)}")
print(f"First sort range: {df['sort1'].min()} to {df['sort1'].max()}")
print(f"Second sort range: {df['sort2'].min()} to {df['sort2'].max()}")
print(f"Number of color groups: {num_groups}")