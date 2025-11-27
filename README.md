# Machine Learning Classification Challenges

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn%20|%20TensorFlow%20|%20CatBoost-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This repository contains the code, analysis, and models for two distinct machine learning classification challenges:
1.  **Startup Founder Retention Prediction** (Binary Classification)
2.  **Multidimensional Personality Cluster Prediction** (Multiclass Classification)

The project explores various algorithms ranging from linear baselines to advanced ensemble methods and neural networks. A key aspect of this study was analyzing model performance under data constraints by training specific models on only **20% of the data** versus the full dataset.

## 👥 Team Members
* **G. Shrujan Teja** - IMT2023599
* **Rajdeep Saha** - IMT2023600
* **Nachiappan.N** - IMT2023605

---

## 📂 Repository Structure

```text
├── Binary_Classification/
│   ├── binary-mlp-new.ipynb           # Best performing Neural Network (Full Data)
│   ├── binary-svm-20-final.ipynb      # SVM trained on 20% data split
│   ├── binary-class-SVCpca.ipynb      # SVM with PCA on Full Data
│   ├── binary-lr-new.ipynb            # Logistic Regression Baseline
│   ├── naive_bayes.ipynb              # Gaussian Naive Bayes
│   └── ml-binary-catboost-1-2.ipynb   # CatBoost Implementation
│
├── Multiclass_Classification/
│   ├── multi-class-mlp.ipynb          # Best performing Neural Network (Full Data)
│   ├── multi-class-svc.ipynb          # SVM on Full Data
│   ├── multi-class-lr.ipynb           # Logistic Regression Baseline
│   ├── ml-pred-svm-multi-class20.ipynb # SVM on 20% data split
│   ├── multi-class-mlp20percent.ipynb  # MLP on 20% data split
│   └── ml-pred-knn.ipynb              # K-Nearest Neighbors
│
├── Reports/
│   └── Project_Report.pdf             # Comprehensive LaTeX report
└── README.md
