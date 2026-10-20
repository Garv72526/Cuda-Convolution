import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# YOUR REAL DATA
hardware = ['CPU (OpenCV/C++)', 'GPU (Global Mem)', 'GPU (Shared Mem Tiled)']
times_ms = [947.05, 1.79, 1.49]

# Set the dark "Hacker" aesthetic
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 6))

# Logarithmic scale because the difference is too huge to see linearly
ax.set_yscale('log')

# Colors: Red for slow, Blue for naive, Neon Green for Tiled
colors = ['#ff4444', '#2196f3', '#00e676']
bars = ax.bar(hardware, times_ms, color=colors, width=0.6)

# Title and Labels
plt.title('Convolution Benchmark: 4K Image (3840x2160)', fontsize=20, fontname='Consolas', color='white', pad=20)
plt.ylabel('Time (ms) - Log Scale', fontsize=12, color='gray')

# Add the text labels above bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        label_y = height * 1.1 if height < 100 else height * 1.05
        ax.text(bar.get_x() + bar.get_width()/2, label_y,
                f'{height} ms',
                ha='center', va='bottom', fontsize=14, fontweight='bold', color='white')

add_labels(bars)

# Add "Speedup" text
speedup = times_ms[0] / times_ms[2]
plt.text(1, 400, f"🚀 {int(speedup)}x Faster than CPU", fontsize=25, color='#00e676', fontweight='bold', ha='center')

# Remove ugly borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.2)

plt.tight_layout()
plt.savefig('benchmark_final.png', dpi=300)
print("Graph Generated: benchmark_final.png")