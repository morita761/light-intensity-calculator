import numpy as np
import matplotlib.pyplot as plt


# Compute pie slices
N = 20
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = 10 * np.random.rand(N)
width = 0.3

ax = plt.subplot(111, projection='polar')
bars = ax.bar(theta, radii, width=width, bottom=0.0)
ax.set_theta_zero_location("S")
# ax.set_theta_direction(-1)
ax.set_xticks(np.radians([0, 90, 180, 270]))
ax.set_xticklabels(['$0^{\circ}$ (下)', '$90^{\circ}$ (右)', '$180^{\circ}$ (上)', '$-90^{\circ}$ (左)'])
ax.set_yticks(np.arange(0, 11, 2))
ax.set_yticklabels(['0', '2', '4', '6', '8', 'MAX (10)'])

# Use custom colors and opacity
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.viridis(r / 10.))
    bar.set_alpha(0.5)

plt.show()