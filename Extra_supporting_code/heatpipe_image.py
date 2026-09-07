# coding=utf-8
import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

data_dir = 'C:/Users/jmayhall/PycharmProjects/PythonProject/files/'

c8_file = None
c13_file = None

# Identify C8 and C13 files
for filename in os.listdir(data_dir):
    if 'C08' in filename:
        c8_file = os.path.join(data_dir, filename)
    elif 'C13' in filename:
        c13_file = os.path.join(data_dir, filename)

if c8_file is None or c13_file is None:
    raise FileNotFoundError("Could not find both C08 and C13 files")

# Open datasets
ds_c8 = xr.open_dataset(c8_file)
ds_c13 = xr.open_dataset(c13_file)

# Extract radiance / brightness temperature
c8 = np.ma.masked_invalid(ds_c8['Rad'].values)
c13 = np.ma.masked_invalid(ds_c13['Rad'].values)

# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)

im0 = axes[0].imshow(c8, cmap='gray_r')
axes[0].set_title('GOES-17 ABI Channel 8', fontsize=20)
axes[0].axis('off')
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).set_label(label=r'$T_b\ (^\circ C)$', fontsize=16)

im1 = axes[1].imshow(c13, cmap='gray_r')
axes[1].set_title('GOES-17 ABI Channel 13', fontsize=20)
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04).set_label(label=r'$T_b\ (^\circ C)$', fontsize=16)

fig.suptitle(
    'GOES-17 ABI Brightness Temperatures\nAugust 7, 2019 – 12 UTC',
    fontsize=24
)

# Save output
output_image = 'c8_c13_side_by_side.png'
plt.savefig(output_image, dpi=300, bbox_inches='tight')
plt.show()

print(f"Image saved as {output_image}")
