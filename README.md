
#  DeepFake Detection Lite  
### Real vs AI-Generated Face Classification using Transfer Learning (EfficientNet, ResNet & MobileNet)

---


## 📌 Project Overview

This project focuses on detecting whether a face image is **real** or **AI-generated (fake)** using deep learning and transfer learning techniques.

---

## 🎯 Goals of the Project

- Classify face images as **Real** or **Fake**
- Extend the dataset using **custom AI-generated fake faces**
- Train and compare multiple CNN architectures:
  - **EfficientNetB0**
  - **ResNet50**
  - **MobileNetV2**
- Evaluate performance using:
  - Accuracy
  - Loss
  - AUC (Area Under Curve)
- Test **generalization** on a completely different dataset (cross-dataset evaluation)

---

## 📂 Dataset Sources

### 🔹 Dataset-A (Training Dataset)
**Real vs Fake Faces Dataset (Kaggle)**  
🔗 https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces  

- 70,000 real images  
- 70,000 fake images  
- Automatically split into:
  - Train
  - Validation
  - Test  
- Balanced class distribution

---

### 🔹 Custom Fake Dataset
Custom AI-generated face images were created using:
- **Gemini Image Generation API**

These images were:
- NOT included in Kaggle dataset
- Used for **unseen evaluation**
- Helpful for testing real-world robustness

---

### 🔹 Dataset-B (Cross-Dataset Evaluation)
**Real and Fake Face Detection Dataset (Kaggle)**  
🔗 https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection  

- Used **only for testing**
- Never used during training
- Helps analyze **domain shift & generalization**


