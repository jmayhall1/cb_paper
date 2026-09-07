# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')

# 0° at top, clockwise increasing
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)

# Degree ticks every 45°
angles = np.arange(0, 360, 45)
ax.set_xticks(np.deg2rad(angles))
ax.set_xticklabels([f"{a}°" for a in angles])

# Hide radial tick labels but keep grid
ax.set_yticklabels([])

# Keep grid lines
ax.grid(True)
plt.tight_layout()
plt.show()
