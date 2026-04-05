import matplotlib.pyplot as plt
import numpy as np
import sum


# Load all 16 by and rg files
by = []
rg = []

for i in range(16):
    # Load by files
    by_data = sum.Data(f"data/by{i}.csv")
    by.append(by_data.sum_spectral_data())
    
    # Load rg files
    rg_data = sum.Data(f"data/rg{i}.csv")
    rg.append(rg_data.sum_spectral_data())

# Create x values: 0th value's x value is -2.2, then -1.2, -0.2, 0.8, etc.
x_values = [-2.2 + i * 1.0 for i in range(16)]

# Plot BY data
plt.figure(figsize=(12, 6))
plt.plot(x_values, by, 'bo-', markersize=8, linewidth=2)
plt.xlabel('Position', fontsize=12)
plt.ylabel('Summed Irradiance (mW/m²/nm)', fontsize=12)
plt.title('BY Measurements', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig('by_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot RG data in separate figure
plt.figure(figsize=(12, 6))
plt.plot(x_values, rg, 'rs-', markersize=8, linewidth=2)
plt.xlabel('Position', fontsize=12)
plt.ylabel('Summed Irradiance (mW/m²)', fontsize=12)
plt.title('RG Measurements', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig('rg_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plots saved as 'by_plot.png' and 'rg_plot.png'")
print(f"X values: {x_values}")
print(f"BY values: {by}")
print(f"RG values: {rg}")
