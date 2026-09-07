# coding=utf-8
"""
@aythor: John Mark Mayhall
Last Edit: 01/31/2026
Code for combining images vertically or horizontally
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.Image import Resampling

if __name__ == "__main__":
    path1 = "C:/Users/jmayhall/Downloads/picture5.png"
    path2 = "C:/Users/jmayhall/Downloads/picture6.png"
    img1 = plt.imread(path1)
    img2 = plt.imread(path2)
    direction = 'vertical'

    if direction == 'vertical':
        # Make widths match if needed
        min_width = min(img1.shape[1], img2.shape[1])
        img1 = img1[:, :min_width, :]
        img2 = img2[:, :min_width, :]

        fig, ax = plt.subplots(figsize=(min_width / 100,
                                        (img1.shape[0] + img2.shape[0]) / 100))

        ax.imshow(np.vstack([img1, img2]))
        ax.axis("off")

        plt.savefig("sst_combined.png", dpi=300, bbox_inches="tight")
        plt.close()

    elif direction == 'horizontal':
        def stack_images_horizontally_scaled(img_paths, output_path, scales=None, gutter=20, dpi=300, bg_color="white"):
            """
            Horizontally stack images with optional per-image scaling.

            Parameters
            ----------
            img_paths : list of str
                Paths to images to stack
            output_path : str
                Where to save the combined image
            scales : list of float, optional
                Scaling factors for each image. Length must match img_paths.
                Default 1.0 for all.
            gutter : int
                Space in pixels between panels
            dpi : int
                Output DPI
            bg_color : str
                Background color
            """
            # Open images
            resample = Resampling.LANCZOS
            imgs = [Image.open(p) for p in img_paths]

            # Apply scaling if provided
            if scales is None:
                scales = [1.0] * len(imgs)
            if len(scales) != len(imgs):
                raise ValueError("Length of scales must match number of images.")

            scaled_imgs = []
            for img, scale in zip(imgs, scales):
                if scale != 1.0:
                    new_width = int(img.width * scale)
                    new_height = int(img.height * scale)
                    scaled_imgs.append(img.resize((new_width, new_height), resample))
                else:
                    scaled_imgs.append(img)

            # Match heights for horizontal stacking
            max_height = max(img.height for img in scaled_imgs)
            padded_imgs = []
            for img in scaled_imgs:
                if img.height < max_height:
                    new_img = Image.new("RGB", (img.width, max_height), color=bg_color)
                    y_offset = (max_height - img.height) // 2
                    new_img.paste(img, (0, y_offset))
                    padded_imgs.append(new_img)
                else:
                    padded_imgs.append(img)

            # Total width including gutters
            total_width = sum(img.width for img in padded_imgs) + gutter * (len(padded_imgs) - 1)

            # Create combined image
            combined = Image.new("RGB", (total_width, max_height), color=bg_color)
            x_offset = 0
            for img in padded_imgs:
                combined.paste(img, (x_offset, 0))
                x_offset += img.width + gutter

            combined.save(output_path, dpi=(dpi, dpi))

        # Usage
        stack_images_horizontally_scaled([path1, path2], "data_maps_combined.png", [.4, 1])