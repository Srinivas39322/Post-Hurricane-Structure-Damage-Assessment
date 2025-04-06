# Core libraries
import os
import warnings
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow / Keras (optional; safely wrapped to avoid crash if not installed)
try:
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.preprocessing import image
    print(f"TensorFlow loaded: v{tf.__version__}")
except ImportError:
    tf = None
    print("TensorFlow is not installed. Skipping related imports.")

# Image processing
from PIL import Image
from skimage import io

# Geospatial
import geopandas as gpd
import contextily as cx
from shapely.geometry import Point
from shapely.errors import GEOSException

# Machine Learning
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity

# IPython magic (only for notebooks)
# %matplotlib inline

# Fix for deprecation warning
np.bool = bool

# Suppress warnings
warnings.filterwarnings("ignore")



# Update directory path based on your folder structure
DIR = "/Users/SRINIVAS/Documents/Capstone Project/Post-hurricane"

folders = ['train_another', 'validation_another', 'test', 'test_another']
for folder in folders:
    folder_path = os.path.join(DIR, folder)
    if not os.path.exists(folder_path):
        print(f"Warning: {folder_path} does not exist.")
    else:
        print(f'-------{folder.upper()}-------')
        for category in ['no_damage', 'damage']:
            category_path = os.path.join(folder_path, category)
            if os.path.exists(category_path):
                count = len(os.listdir(category_path))
                print(f"{category.capitalize()}: {count}")
            else:
                print(f"Warning: {category_path} does not exist.")

def load_image(fp, size=(128,128)):
    img = Image.open(fp).convert('RGB').resize(size)
    return np.array(img)

train_dir = os.path.join(DIR, 'train_another')
if os.path.exists(train_dir):
    damage_imgs = [fn for fn in os.listdir(f'{train_dir}/damage') if fn.endswith('.jpeg')]
    no_damage_imgs = [fn for fn in os.listdir(f'{train_dir}/no_damage') if fn.endswith('.jpeg')]

    damage_df = pd.DataFrame(damage_imgs, columns=['Filename'])
    damage_df[['lon', 'lat']] = damage_df['Filename'].str.replace('.jpeg', '', regex=False).str.split('_', expand=True)
    damage_df['damage'] = 1

    no_damage_df = pd.DataFrame(no_damage_imgs, columns=['Filename'])
    no_damage_df[['lon', 'lat']] = no_damage_df['Filename'].str.replace('.jpeg', '', regex=False).str.split('_', expand=True)
    no_damage_df['damage'] = 0

    all_df = pd.concat([damage_df, no_damage_df])
    all_df['lon'] = pd.to_numeric(all_df['lon'], errors='coerce')
    all_df['lat'] = pd.to_numeric(all_df['lat'], errors='coerce')

    all_df = all_df.dropna(subset=['lon', 'lat']).drop_duplicates()


#load a sample image
tile_path = os.path.join(DIR, 'train_another', 'no_damage' , '-95.086_29.827665000000003.jpeg')
tile = io.imread(tile_path)

#plot image
plt.figure(figsize=(5,5))
plt.imshow(tile);


import os

path = "/Users/SRINIVAS/Documents/Capstone Project/Post-hurricane/train_another/damage"
try:
    print("Files:", os.listdir(path))
except PermissionError:
    print("Permission denied for:", path)

import tensorflow as tf
import os

# Define directory and parameters
DIR = "/Users/SRINIVAS/Documents/Capstone Project/Post-hurricane"
image_size = (128, 128)
batch_size = 32
seed = 42

train_dir = os.path.join(DIR, 'train_another')
val_dir = os.path.join(DIR, 'validation_another')

# Step 1: Check for corrupted images
num_skipped = 0
for folder_name in ("damage", "no_damage"):
    folder_path = os.path.join(train_dir, folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            with open(fpath, "rb") as fobj:
                is_jfif = b"JFIF" in fobj.peek(10)
        except Exception as e:
            print(f"Error reading file {fpath}: {e}")
            is_jfif = False

        if not is_jfif:
            num_skipped += 1
            print(f"Corrupted: {fpath}")
            # os.remove(fpath)  # Uncomment to delete

print(f"\nTotal corrupted images found: {num_skipped}")

# Step 2: Load datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='binary'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
    label_mode='binary'
)

# Step 3: Normalize and prefetch datasets
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# 🔁 Set the base image directory
image_dir = "/Users/SRINIVAS/Documents/Capstone Project/Post-hurricane/train_another"

# 📂 Collect image metadata
image_data = []

for label in ["damage", "no_damage"]:
    folder = os.path.join(image_dir, label)
    for fname in os.listdir(folder):
        if fname.endswith(".jpeg") or fname.endswith(".jpg") or fname.endswith(".png"):
            try:
                # Example: "-93.6141_30.754263.jpeg"
                lon_str, lat_str = fname.replace(".jpeg", "").replace(".jpg", "").replace(".png", "").split("_")
                lon, lat = float(lon_str), float(lat_str)
                image_data.append({"filename": fname, "label": label, "lat": lat, "lon": lon})
            except Exception as e:
                print(f"❌ Skipping invalid filename: {fname} ({e})")

# 🧾 Create DataFrame
all_df = pd.DataFrame(image_data)

# 🌍 Create GeoDataFrame
try:
    all_df['geometry'] = all_df.apply(
        lambda row: Point(row['lon'], row['lat']) if pd.notnull(row['lon']) and pd.notnull(row['lat']) else None,
        axis=1
    )
    all_df = all_df.dropna(subset=['geometry'])

    all_gdf = gpd.GeoDataFrame(all_df, geometry='geometry', crs='EPSG:4326').to_crs(epsg=3857)
    print("✅ GeoDataFrame created successfully!")
    print(all_gdf.head())

except Exception as e:
    print(f"❌ Error creating GeoDataFrame: {e}")


import os
import numpy as np
import pandas as pd
from PIL import Image

# Set directory
train_dir = "/Users/SRINIVAS/Documents/Capstone Project/Post-hurricane/train_another"

# Function to load and flatten images
def load_and_flatten_image(fp, size=(64, 64)):
    img = Image.open(fp).convert('RGB').resize(size)
    return np.array(img).flatten()

# Get image filenames
damage_dir = os.path.join(train_dir, 'damage')
no_damage_dir = os.path.join(train_dir, 'no_damage')

damage_imgs = [img for img in os.listdir(damage_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
no_damage_imgs = [img for img in os.listdir(no_damage_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]

# Get file paths (limit to 100 each)
damage_paths = [os.path.join(damage_dir, img) for img in damage_imgs[:100]]
no_damage_paths = [os.path.join(no_damage_dir, img) for img in no_damage_imgs[:100]]

# Combine and label
image_data = []
labels = []

for path in damage_paths:
    image_data.append(load_and_flatten_image(path))
    labels.append(1)

for path in no_damage_paths:
    image_data.append(load_and_flatten_image(path))
    labels.append(0)

# Convert to DataFrame
pixel_df = pd.DataFrame(image_data)
pixel_df['label'] = labels

print("✅ Pixel DataFrame created successfully!")
print(pixel_df.shape)
print(pixel_df.head())


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Ensure pixel_df exists and 'label' column is present
X = pixel_df.drop('label', axis=1)
y = pixel_df['label']

# Run PCA to reduce dimensionality to 2D
pca = PCA(n_components=2)
components = pca.fit_transform(X)

# Plot PCA results
plt.figure(figsize=(8, 6))
sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=y, palette='coolwarm', s=60, alpha=0.7)
plt.title("PCA of Flattened Image Pixels")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

## Pixel Intensity Distribution
damage_pixels = pixel_df[pixel_df['label'] == 1].drop('label', axis=1).mean()
no_damage_pixels = pixel_df[pixel_df['label'] == 0].drop('label', axis=1).mean()

plt.figure(figsize=(10, 4))
plt.plot(damage_pixels.values[:1000], label='Damage', alpha=0.7)
plt.plot(no_damage_pixels.values[:1000], label='No Damage', alpha=0.7)
plt.legend()
plt.title("Average Pixel Intensity Comparison (First 1000 Pixels)")
plt.xlabel("Pixel Index")
plt.ylabel("Average Intensity")
plt.show()

# Heatmap of Mean Images
import matplotlib.pyplot as plt
import numpy as np

# Reuse your function or define it inline
def find_mean_img(full_mat, size=(128, 128)):
    return np.mean(full_mat, axis=0).reshape(size)

# Compute mean images
damage_mean = find_mean_img(damage_arrays, size=(128, 128))
no_damage_mean = find_mean_img(no_damage_arrays, size=(128, 128))

# === Plot both with consistent vmin/vmax and plasma colormap ===
fig, axs = plt.subplots(2, 1, figsize=(6, 10))

# DAMAGE heatmap
im1 = axs[0].imshow(damage_mean, cmap='plasma', vmin=85, vmax=110)
axs[0].set_title("Average DAMAGE")
axs[0].axis('off')
plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

# NO DAMAGE heatmap
im2 = axs[1].imshow(no_damage_mean, cmap='plasma', vmin=85, vmax=110)
axs[1].set_title("Average NO DAMAGE")
axs[1].axis('off')
plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()


# Standard Deviation by Class

import matplotlib.pyplot as plt
import numpy as np

# === Step 1: Compute Standard Deviation Images ===
damage_std = np.std(damage_arrays, axis=0).reshape((128, 128))
no_damage_std = np.std(no_damage_arrays, axis=0).reshape((128, 128))

# === Step 2: Plot both side-by-side with consistent color scale and colormap ===
fig, axs = plt.subplots(2, 1, figsize=(6, 10))

# Plot DAMAGE standard deviation
im1 = axs[0].imshow(damage_std, cmap='plasma', vmin=20, vmax=70)
axs[0].set_title("Standard Deviation DAMAGE")
axs[0].axis('off')
plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

# Plot NO-DAMAGE standard deviation
im2 = axs[1].imshow(no_damage_std, cmap='plasma', vmin=20, vmax=70)
axs[1].set_title("Standard Deviation NO DAMAGE")
axs[1].axis('off')
plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# Contrast Between Average Images

import matplotlib.pyplot as plt
import numpy as np

# === Step 1: Compute mean images ===
damage_mean = np.mean(damage_arrays, axis=0).reshape((128, 128))
no_damage_mean = np.mean(no_damage_arrays, axis=0).reshape((128, 128))

# === Step 2: Calculate raw difference ===
diff_img = damage_mean - no_damage_mean

# === Step 3: Plot the difference heatmap ===
plt.figure(figsize=(6, 5))
plt.imshow(diff_img, cmap='seismic', vmin=-15, vmax=15)
plt.title("Difference Between Damage & No Damage Average")
plt.axis('off')
plt.colorbar(label="Pixel Difference")
plt.tight_layout()
plt.show()

# Eigenimages

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# === Step 1: Stack all images into a matrix (grayscale, flattened)
all_images = np.vstack([damage_arrays, no_damage_arrays])  # Shape: (num_images, 128*128)

# === Step 2: Perform PCA
num_components = 56
pca = PCA(n_components=num_components)
pca.fit(all_images)

# === Step 3: Extract eigenvectors (principal components)
eigenimages = pca.components_.reshape((num_components, 128, 128))

# === Step 4: Plot eigenimages
fig, axs = plt.subplots(7, 8, figsize=(12, 10))  # Adjust grid based on num_components
axs = axs.flatten()

for i in range(num_components):
    axs[i].imshow(eigenimages[i], cmap='gray')
    axs[i].axis('off')

# If fewer axes than grid, hide the rest
for j in range(num_components, len(axs)):
    axs[j].axis('off')

plt.suptitle(f'Number of PC: {num_components}', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

# Geographic Distribution
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

# Project to Web Mercator
all_gdf_web = all_gdf.to_crs(epsg=3857)

# Define color mapping for labels (assuming string values)
colors = {'no_damage': 'blue', 'damage': 'red'}
labels = {'no_damage': 'No Damage', 'damage': 'Damage'}

# Create plot
fig, ax = plt.subplots(figsize=(12, 10))

# Plot points by label
for label_val, group in all_gdf_web.groupby('label'):
    group.plot(
        ax=ax,
        markersize=8,
        color=colors[label_val],
        label=labels[label_val],
        alpha=0.7
    )

# ✅ Add Esri satellite basemap
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldTopoMap)

# Final touches
ax.set_title("Point locations of Training Images, Damaged or Not", fontsize=16)
ax.legend()
ax.axis('off')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")

# Create side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
fig.suptitle("Distribution of Longitudes", fontsize=16)

# Plot No Damage
sns.histplot(all_gdf[all_gdf['label'] == 'no_damage']['lon'], ax=ax1, color='blue', bins=40)
ax1.set_xlim([-97, -93.5])
ax1.set_title('No Damage')
ax1.set_xlabel('Longitude')

# Plot Damage
sns.histplot(all_gdf[all_gdf['label'] == 'damage']['lon'], ax=ax2, color='red', bins=40)
ax2.set_xlim([-97, -93.5])
ax2.set_title('Damage')
ax2.set_xlabel('Longitude')

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

import matplotlib.pyplot as plt
import contextily as cx

# Copy and map label to numeric
all_gdf_copy = all_gdf.copy()
all_gdf_copy['label_numeric'] = all_gdf_copy['label'].map({'no_damage': 0, 'damage': 1})

# Filter for southwestern area and convert CRS
southern_gdf = all_gdf_copy[all_gdf_copy['lon'] < -96].to_crs(epsg=3857)

# Plot
fig, ax = plt.subplots(figsize=(10, 10))

southern_gdf.plot(
    ax=ax,
    column='label_numeric',
    categorical=True,
    legend=True,
    alpha=0.8,
    markersize=1.5,
    cmap='bwr'  # 0: blue, 1: red
)

# Set bounding box (Victoria / Cuero / Port Lavaca region)
ax.set_xlim([-10798210.908 - 60000, -10782077.708 + 60000])

# ✅ Use hybrid basemap with place names
cx.add_basemap(ax, source=cx.providers.Esri.WorldTopoMap)

# Title & final touches
ax.set_title('Point locations of Training Images, Damaged or Not (Southwestern Grouping) - Victoria', fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.show()


import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define image directory
image_dir = "/Users/SRINIVAS/Documents/Capstone Project/Post-hurricane/train_another"

# Build full path to each image
all_gdf['file_path'] = all_gdf.apply(
    lambda row: os.path.join(image_dir, row['label'], row['filename']),
    axis=1
)


valid_gdf['file_size_kb'] = valid_gdf['file_path'].apply(lambda x: os.path.getsize(x) / 1024)

# Preview actual file sizes
print(valid_gdf[['file_path', 'file_size_kb']].head())


valid_gdf['damage'] = valid_gdf['label'].map({'no_damage': 0, 'damage': 1})

plt.figure(figsize=(6, 5))
sns.boxplot(data=valid_gdf, x='damage', y='file_size_kb', palette='Set2')

plt.xlabel("damage")
plt.ylabel("file_size (KB)")
plt.title("File Size Distribution by Damage Class")
plt.tight_layout()
plt.show()


all_gdf

# Canny Edge Detection for Damage Feature Extraction

import cv2

img_path = '/Users/SRINIVAS/Documents/Capstone Project/Post-hurricane/train_another/damage/-93.6141_30.754263.jpeg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


if img is None:
    print("Image not found or invalid path:", img_path)
else:
    edges = cv2.Canny(img, threshold1=100, threshold2=200)
    plt.imshow(edges, cmap='gray')
    plt.title("Canny Edge Detection")
    plt.axis('off')
    plt.show()


## GLCM-Based Texture Feature Extraction

from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
import numpy as np

# Load and convert image
img = imread(img_path)
gray = rgb2gray(img)
gray_u8 = img_as_ubyte(gray)  # Convert to uint8

# Compute GLCM
glcm = graycomatrix(gray_u8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

# Extract features
contrast = graycoprops(glcm, 'contrast')[0, 0]
correlation = graycoprops(glcm, 'correlation')[0, 0]
energy = graycoprops(glcm, 'energy')[0, 0]
homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

print("GLCM Features → Contrast:", contrast, "| Correlation:", correlation)

