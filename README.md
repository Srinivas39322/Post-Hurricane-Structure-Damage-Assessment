![Hurricane Damage](Hurricane_Damage.jpeg)

# Post-Hurricane-Structure-Damage-Assessment
We plan to deploy three models: CNN, MobileNet, and ResNet-50, with a focus on using transfer learning techniques. These techniques enable the models to leverage large pre-trained datasets, facilitating a more generalized and efficient learning process that is well-suited to categorizing satellite imagery where data may be less detailed.

# Hurricane Impact Analysis System 🌪️
**Group 8: Tenneti Srinivas Saiteja | Namratha Prakash | Lakshmi Sreya Rapolu**  
*A Machine Learning Approach to Satellite Image Analysis*

---

![Project Overview](image.png)

## 📌 Project Overview
This project aims to develop a machine learning system to classify hurricane-damaged buildings from satellite images. By combining CNN architectures with domain-specific image processing techniques, our system enables rapid post-disaster damage assessments.

---

## 📂 Contents
1. [Data Acquisition & Preprocessing](#1-data-acquisition-and-preprocessing)
2. [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
3. [Feature Engineering](#3-feature-engineering)
4. [Data Augmentation](#4-data-augmentation)
5. [Model Training](#6-model-selection-and-training)
6. [Evaluation](#7-evaluation-and-iteration)
7. [Web App Deployment](#8-web-application-development-for-deployment)
8. [Cloud Infrastructure](#9-cloud-deployment-and-management)

---

## 1. Data Acquisition and Preprocessing
- Accessed datasets using Python libraries like NumPy and PIL.
- Resized all images to `128x128` pixels for uniformity.

## 2. Exploratory Data Analysis (EDA)
- Performed visual inspection and class distribution analysis.
- Assessed image quality to detect potential classification challenges.

## 3. Feature Engineering
- Scaled pixel values to `[0, 1]`.
- Applied edge detection (Sobel, Canny) and texture analysis (GLCM).
- Implemented PCA for dimensionality reduction and watershed segmentation.

## 4. Data Augmentation
- Used PyTorch to apply:
  - Rotation
  - Zooming
  - Horizontal flipping  
This enriched the dataset for better model generalization.

## 5. Data Preparation
- **Training:** `train_another` — 5,000 images/class  
- **Validation:** `validation_another` — 1,000 images/class  
- **Test (Unbalanced):** 8,000 damaged | 1,000 undamaged  
- **Test (Balanced):** 1,000 per class  
- Batched using `PyTorch DataLoader`.

## 6. Model Selection and Training
- Fine-tuned CNN architectures to handle `128x128` inputs.
- Leveraged **MobileNet** and **ResNet-50** with transfer learning.
- Integrated decaying learning rate schedules to improve convergence.

## 7. Evaluation and Iteration
- Metrics used: `accuracy`, `precision`, `recall`, `F1-score`.
- Continuous refinement of models and preprocessing pipelines.

## 8. Web Application Development for Deployment
- **Backend:** Flask for Python model integration.
- **Frontend:** HTML, CSS, JS (with potential React/Angular).
- **UI:** Dashboard for image upload, prediction results, and damage maps.

## 9. Cloud Deployment and Management
- **Platform:** AWS (EC2, S3, Elastic Beanstalk).
- **Focus:** Scalability and high availability during disaster response.

---

## 🚀 Future Work
- Add support for multi-class damage levels.
- Integrate GIS for geo-tagged predictions.
- Expand to include fire, flood, or earthquake scenarios.

---

## 📸 Sample Visuals
(Place additional model output or EDA images here if available)

---

## 📬 Contact
For questions, contact any team member via this repository's issue tracker.

