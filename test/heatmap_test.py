import numpy as np
import matplotlib.pyplot as plt

# Grid centered around one point
grid_size = 101
extent = 3.0
x = np.linspace(-extent, extent, grid_size)
y = np.linspace(-extent, extent, grid_size)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)

sigma = 1.0
radius = 2.0

kernels = {
    "Gaussian": np.exp(-0.5 * (R / sigma) ** 2),

    "Epanechnikov": np.where(
        R <= radius,
        1 - (R / radius) ** 2,
        0
    ),

    "Uniform / Disk": np.where(
        R <= radius,
        1,
        0
    ),

    "Triangular / Cone": np.where(
        R <= radius,
        1 - R / radius,
        0
    ),

    "Exponential": np.exp(-R / sigma),

    "Laplacian": np.exp(-R / sigma),

    "Cauchy / Lorentzian": 1 / (1 + (R / sigma) ** 2),

    "Biweight": np.where(
        R <= radius,
        (1 - (R / radius) ** 2) ** 2,
        0
    ),

    "Tricube": np.where(
        R <= radius,
        (1 - (R / radius) ** 3) ** 3,
        0
    ),
}

# Normalize for visual comparison
kernels = {name: arr / arr.max() for name, arr in kernels.items()}

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes = axes.ravel()

for ax, (name, heatmap) in zip(axes, kernels.items()):
    im = ax.imshow(
        heatmap,
        extent=[-extent, extent, -extent, extent],
        origin="lower",
        cmap="hot",
        vmin=0,
        vmax=1
    )
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])

# Shared colorbar


plt.suptitle("2D Heatmap Kernels Centered at a Single Point", fontsize=14)
plt.tight_layout()
plt.show()