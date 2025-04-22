![Hurricane Damage](Hurricane_Damage.jpeg)

# 🌪️ Post-Hurricane Damage Detection with Deep Learning

A machine learning-powered solution for rapid post-disaster assessment using aerial imagery. This project leverages deep learning models—ResNet50, CNN, and EfficientNet—to classify structural damage caused by hurricanes. Designed for real-time predictions via a Streamlit web interface.

---

## 👩‍💻 Team Members
- **Srinivas Saiteja Tenneti**
- **Namratha Prakash**
- **Lakshmi Sreya Rapolu**

---

## 📌 Project Overview

Hurricanes in the U.S. cause an average of **$21.5 billion** in damage per event, with over 10 billion-dollar storms annually between 2015 and 2020. Accurate and rapid **post-hurricane damage assessment** is essential for emergency response, insurance processing, and recovery planning.

This project builds an AI system that:
- Detects structural damage from **aerial images** post-hurricane
- Utilizes **transfer learning** with **ResNet50 and EfficientNet** models
- Deploys a **Streamlit web app** for interactive image uploads and predictions

---

## 🛰️ Dataset

- **Source**: University of Washington Disaster Data Science Lab
- **Location**: Houston, TX (Post Hurricane Harvey)
- **Images**: 14,000 (7,000 damaged, 7,000 undamaged)
- **Splits**:
  - `Train`: 8,000
  - `Validation`: 2,000
  - `Test`: 2,000 (also tested on unbalanced and balanced subsets)

---

## 🛠️ Key Techniques Used

- Data Normalization & Augmentation (`RandomHorizontalFlip`)
- PCA for feature reduction
- Custom & pre-trained models
- Evaluation metrics: Accuracy, Confusion Matrix, F1-score
- Streamlit-based real-time interface for multi-image upload

---

## 📊 Exploratory Data Analysis (EDA)

- **Pixel Mean & Std Dev** revealed subtle texture differences between damaged and undamaged classes.
- **PCA**:
  - Damaged: 70% variance in just 19 components
  - Undamaged: Needed 56 components
- **Pixel Intensity**: Damaged areas tend to be darker and more uniform.
- **Geospatial Bias**: Model risks learning location-based patterns — spatial regularization needed.

---


## 🧠 Model Highlights

### 🏗️ Custom CNN (from scratch)
- Input: `128x128 RGB images`
- Architecture: 3 Conv Layers + 4 FC Layers
- Accuracy:
  - ✅ Train: 99.48%
  - ✅ Validation: 96.25%

---

### 🦾 ResNet50 (Transfer Learning)
- Input: `224x224`, ImageNet normalized
- Accuracy:
  - ✅ Validation: **99.50%**
  - ✅ Test Set: **99.61%**
- 🧠 Best model for generalization and deployment

---

### 🌱 EfficientNet Models
#### EfficientNet-B0:
- Accuracy: 99.30%
- Lightweight and fast but slightly underperformed vs. ResNet50

#### EfficientNet-V2-S:
- **Frozen:** 91.7% accuracy — very fast but limited learning
- **Fine-tuned (Last 2 Blocks):** 97.95% — efficient and effective

---

## 💻 Streamlit Web App

Interactive interface for uploading and classifying images.

### Features:
- Multi-image upload with grid view
- Class predictions (damage / no damage)
- Confidence scores with visual indicators
- Session-wise prediction history
- Optional visualization of transformed model input
- Lightweight and runs locally or on any Streamlit-compatible server

---

## 📈 Sample Result Snapshot

- ✅ **ResNet50 Confusion Matrix**
  - True Positives: 7,980
  - False Negatives: 20
  - True Negatives: 985
  - False Positives: 15
  - Accuracy: **99.61%**

---

## 🚀 Future Work

- Multi-class damage levels (minor/moderate/severe)
- Integrate Grad-CAM for visual attention maps
- Expand to detect other disaster types: fire, floods, earthquakes
- Incorporate geospatial overlays using GIS libraries

---

## 📷 Sample Visuals (Coming Soon)

- Eigenimages from PCA
- Geospatial distribution heatmaps
- Grad-CAM overlays (model attention)

---

## 📬 Contact

For questions or contributions, reach out via GitHub Issues or connect with the team:

- **[Srinivas Saiteja Tenneti](https://www.linkedin.com/in/srinivas-saiteja-tenneti/)**

---

## 🔗 References
- [Dataset on IEEE Dataport](https://ieee-dataport.org/open-access/detecting-damaged-buildings-post-hurricane-satellite-imagery-based-customized)
- [University of Washington Disaster Data Science Lab](https://disasterdatascience.org/)

---

> “In the aftermath of a hurricane, every second counts. With AI-driven tools, response teams can act faster and smarter.” – Group 8


## 📬 Contact
For questions, contact any team member via this repository's issue tracker.

