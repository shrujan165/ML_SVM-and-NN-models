# Machine Learning Classification Challenges

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn%20|%20TensorFlow%20|%20CatBoost-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This repository contains the code, analysis, and predictive models for two distinct machine learning classification competitions. The project explores the performance of various algorithms—ranging from linear baselines to advanced ensemble methods and neural networks—under different data constraints (20% vs. 100% training data).

### The Challenges
1.  **Startup Founder Retention Prediction** (Binary Classification)
2.  **Multidimensional Personality Cluster Prediction** (Multiclass Classification)

---

## 👥 Team Members
* **G. Shrujan Teja** - IMT2023599
* **Rajdeep Saha** - IMT2023600
* **Nachiappan.N** - IMT2023605

---

## 📂 Repository Structure

### Binary Classification (Founder Retention)
* `binary-mlp-new.ipynb`: Neural Network trained on full data (Best Model).
* `binary-svm-20-final.ipynb`: SVM trained on 20% data split.
* `binary-class-SVCpca.ipynb`: SVM with PCA trained on full data.
* `binary-lr-new.ipynb`: Logistic Regression baseline.
* `naive_bayes.ipynb`: Gaussian Naive Bayes implementation.
* `ml-binary-catboost-1-2.ipynb`: CatBoost implementation.

### Multiclass Classification (Personality Clusters)
* `multi-class-mlp.ipynb`: Neural Network trained on full data (Best Model).
* `multi-class-svc.ipynb`: SVM trained on full data.
* `multi-class-lr.ipynb`: Logistic Regression baseline.
* `ml-pred-svm-multi-class20.ipynb`: SVM trained on 20% data split.
* `multi-class-mlp20percent.ipynb`: Neural Network trained on 20% data split.
* `ml-pred-knn.ipynb`: K-Nearest Neighbors implementation.

### Reports
* `Project_Report.pdf`: Comprehensive project report with methodology and analysis.

---

## 🚀 Task 1: Startup Founder Retention (Binary)

### Dataset Description
**Goal:** Predict whether a startup founder will stay with (`Retained`) or leave (`Exited`) their venture.
* **Target:** `retention_status`
* **Key Features:** `founder_age`, `monthly_revenue_generated`, `years_with_startup`, `work_life_balance_rating`.

### Methodology
* **Feature Engineering:**
    * **Burnout Index:** Interaction feature between `working_overtime` and `work_life_balance_rating`.
    * **Log Transformation:** Applied to skewed numerical features like revenue.
* **Models:** Comparison of Logistic Regression, SVM, Naive Bayes, CatBoost, and MLP.

---

## 🧠 Task 2: Personality Cluster Prediction (Multiclass)

### Dataset Description
**Goal:** Classify individuals into one of 5 personality clusters based on behavioral data.
* **Target:** `personality_cluster` (Clusters A-E)
* **Key Features:** `focus_intensity`, `consistency_score`, `hobby_engagement_level`.
* **Metric:** Macro F1 Score (due to class imbalance).

### Methodology
* **Feature Engineering:**
    * **Total Engagement:** Aggregated score of hobbies, physical activity, and altruism.
    * **Consistency Ratio:** Ratio of consistency score to focus intensity.
* **Models:** Comparison of Logistic Regression, KNN, SVM, and MLP.

---

## 🏆 Kaggle Leaderboard Results

We evaluated all models on the Kaggle Public Leaderboard. Below are the final scores.

### Binary Classification Results
| Rank | Model | Data Used | Accuracy Score |
| :--- | :--- | :---: | :---: |
| **1** | **Neural Network (MLP)** | **100%** | **0.747** |
| 2 | CatBoost | 100% | 0.745 |
| 3 | Naive Bayes | 100% | 0.743 |
| 4 | Neural Network (MLP) | 20% | 0.740 |
| 5 | SVM | 20% | 0.735 |
| 6 | SVM | 100% | 0.734 |
| 7 | Logistic Regression | 100% | 0.727 |

### Multiclass Classification Results
| Rank | Model | Data Used | Macro F1 Score |
| :--- | :--- | :---: | :---: |
| **1** | **Neural Network (MLP)** | **100%** | **0.641** |
| 2 | SVM | 100% | 0.552 |
| 3 | Logistic Regression | 100% | 0.541 |
| 4 | Neural Network (MLP) | 20% | 0.530 |
| 5 | SVM | 20% | 0.514 |
| 6 | K-Nearest Neighbors (KNN) | 100% | 0.435 |

---

## 🔑 Key Findings

1.  **Neural Networks Superiority:** In both tasks, the MLP (Multi-Layer Perceptron) achieved the highest scores. This was especially pronounced in the Multiclass task, where it outperformed SVM by nearly **9%**, indicating strong non-linear relationships in the personality data.
2.  **Impact of Data Size:**
    * **Binary Task:** The drop in performance when using only 20% of data was minimal (0.747 $\to$ 0.740), suggesting the dataset had strong linear separability that could be learned quickly.
    * **Multiclass Task:** The drop was significant (0.641 $\to$ 0.530), highlighting that complex multiclass boundaries require significantly more data to generalize well.
3.  **Feature Importance:** The engineered `burnout_index` was a critical predictor for founder retention, significantly boosting the performance of tree-based models like CatBoost.

---

## 🛠️ Requirements
To run the notebooks, install the following dependencies:

```bash
pip install pandas numpy scikit-learn xgboost catboost tensorflow seaborn matplotlib
