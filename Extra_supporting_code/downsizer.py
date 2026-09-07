# coding=utf-8
import os
from PIL import Image

folder = r"C:\Users\jmayhall\Downloads\extended_abstract"

# settings
max_dimension = 2500   # resize if width or height exceeds this
jpeg_quality = 25      # compression quality (lower = smaller file)

for filename in os.listdir(folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):

        path = os.path.join(folder, filename)

        try:
            with Image.open(path) as img:

                width, height = img.size

                # Resize if image is very large
                if max(width, height) > max_dimension:
                    scale = max_dimension / max(width, height)
                    new_size = (int(width * scale), int(height * scale))
                    img = img.resize(new_size, Image.LANCZOS)

                # Save compressed
                if filename.lower().endswith(".png"):
                    img.save(path, optimize=True)

                else:  # jpeg
                    img = img.convert("RGB")
                    img.save(path, quality=jpeg_quality, optimize=True)

                print(f"Compressed: {filename}")

        except Exception as e:
            print(f"Failed: {filename} ({e})")

print("Done.")