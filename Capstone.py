#%%
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image, ImageFilter

def is_blurry(image, threshold=100.0):
    gray = image.convert('L')
    variance = np.var(gray.filter(ImageFilter.FIND_EDGES))
    return variance < threshold

def clean_image_dataset(source_root, cleaned_root, target_size=(256, 256), blur_threshold=100.0):
    os.makedirs(cleaned_root, exist_ok=True)

    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                file_path = os.path.join(root, file)
                try:
                    image = Image.open(file_path)
                except:
                    continue

                if is_blurry(image, threshold=blur_threshold):
                    continue

                image_resized = image.resize(target_size)

                relative_path = os.path.relpath(root, source_root)
                save_dir = os.path.join(cleaned_root, relative_path)
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, file)
                image_resized.save(save_path)

    print(f"Dataset cleaning and separation completed. Cleaned data stored in: {cleaned_root}")
# Example usage:
source_directory = "/Users/SRINIVAS/Desktop/Capstone Project/Post-hurricane"
cleaned_directory = "/Users/SRINIVAS/Desktop/Capstone Project/Cleaned-Post-hurricane"
clean_image_dataset(source_directory, cleaned_directory, target_size=(256, 256), blur_threshold=120.0)

# %%

