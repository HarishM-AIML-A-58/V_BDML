# 🧠 Build and Deployment of Machine Learning Applications
**Course Code:** AI23521 | **Semester:** V | **Year:** 2025–2026  
**Author:** Harish M (Reg. No. 2116-231501058)  
**Institution:** Rajalakshmi Engineering College, Chennai  

---

## 📋 Table of Contents
| Section | Description |
|----------|-------------|
| [📘 Introduction](#-introduction) | Overview of the repository |
| [🧩 Experiments Overview](#-experiments-overview) | List of all implemented experiments |
| [⚙️ Technologies Used](#️-technologies-used) | Tools, frameworks, and environments |
| [📊 Experiment Details](#-experiment-details) | Step-by-step breakdown of each experiment |
| [🚀 Deployment](#-deployment) | Instructions for containerization & API deployment |
| [🧠 Learning Outcomes](#-learning-outcomes) | Skills gained from each module |
| [📂 Repository Structure](#-repository-structure) | Folder and file organization |
| [🏁 Conclusion](#-conclusion) | Final remarks and outcomes |
| [👨‍💻 Author](#-author) | Author details |

---

## 📘 Introduction
This repository is a complete record of the **Build and Deployment of Machine Learning Applications** laboratory course, focusing on implementing, optimizing, and deploying ML models.  
It includes foundational preprocessing, classical ML, deep learning, generative models, and deployment workflows using **Flask + Docker**.

---

## 🧩 Experiments Overview
| Exp No. | Title | Key Concepts |
|----------|--------|--------------|
| 1 | Setting Up the Environment and Preprocessing the Data | Data Cleaning, Encoding, Scaling |
| 2 | SVM and Random Forest for Binary & Multiclass Classification | Model Training, Evaluation |
| 3 | Classification with Decision Trees | Gini Index, Visualization |
| 4A | Support Vector Machines (SVM) | Hyperparameter Tuning, ROC-AUC |
| 4B | Ensemble Methods: Random Forest | Feature Importance, Bagging |
| 5 | Clustering with K-Means & PCA | Unsupervised Learning, Dimensionality Reduction |
| 6 | Feedforward & Convolutional Neural Networks | Deep Learning using Keras |
| 7 | Generative Models with GANs | Synthetic Data Generation |
| 8 | Hyperparameter Tuning & Cross-Validation | Grid Search, k-Fold CV |
| 9 | Mini Project – Stock Price Forecasting | Time Series, Linear Regression, Flask Deployment |

---

## ⚙️ Technologies Used
| Category | Tools |
|-----------|-------|
| Programming Language | Python 3.11 |
| ML Libraries | scikit-learn, tensorflow, keras, pandas, numpy, matplotlib, seaborn |
| Deployment | Flask, Docker |
| Visualization | matplotlib, seaborn, plotly |
| Data | Iris, MNIST, Titanic, Breast Cancer, Synthetic Datasets |
| IDEs | Jupyter Notebook, VS Code |

---

## 📊 Experiment Details
### Experiment 1: Environment Setup & Preprocessing
- **Goal:** Set up ML environment and preprocess dataset  
- **Highlights:** Missing value imputation, label encoding, feature scaling  

### Experiment 2: SVM & Random Forest
- **Goal:** Apply SVM and Random Forest for classification tasks  
- **Concepts:** Binary & Multiclass Classification, Accuracy, Confusion Matrix  

### Experiment 3: Decision Trees
- **Goal:** Implement decision tree classifier  
- **Tools:** DecisionTreeClassifier, plot_tree()  

### Experiment 4A & 4B: SVM and Random Forest (Advanced)
- **Focus:** Hyperparameter Tuning, GridSearchCV, Feature Importance  

### Experiment 5: K-Means & PCA
- **Goal:** Cluster data and reduce dimensionality  
- **Methods:** Elbow Method, Silhouette Score, PCA Visualization  

### Experiment 6: FNN & CNN
- **Goal:** Implement deep learning architectures using Keras  
- **Dataset:** Fashion-MNIST, MNIST  

### Experiment 7: Generative Adversarial Networks
- **Goal:** Generate synthetic images using GANs  
- **Concepts:** Generator–Discriminator training loop  

### Experiment 8: Hyperparameter Tuning & Cross Validation
- **Goal:** Optimize SVM with Grid Search and validate with k-Fold CV  

### Experiment 9: Mini Project — Stock Price Forecasting
- **Goal:** Forecast stock prices using Linear Regression  
- **Stack:** Flask + HTML + Bootstrap  

---

## 🚀 Deployment
| Step | Description |
|------|-------------|
| 1 | Build Flask App — `python app.py` |
| 2 | Containerize with Docker — Create Dockerfile and build image |
| 3 | Run Container — `docker run -p 5000:5000 bdml-app` |
| 4 | Access App — Visit http://localhost:5000 |

---

## 🧠 Learning Outcomes
- Mastered ML workflow from preprocessing to deployment  
- Applied deep learning models (CNNs, GANs)  
- Understood model evaluation & optimization (Grid Search, CV)  
- Built and containerized real-world applications using Flask + Docker  

---

## 📂 Repository Structure
```
📦 BDML-Lab
 ┣ 📂 Experiment_1_Preprocessing/
 ┣ 📂 Experiment_2_SVM_RandomForest/
 ┣ 📂 Experiment_3_DecisionTree/
 ┣ 📂 Experiment_4_SVM_RF_Tuning/
 ┣ 📂 Experiment_5_KMeans_PCA/
 ┣ 📂 Experiment_6_FNN_CNN/
 ┣ 📂 Experiment_7_GAN/
 ┣ 📂 Experiment_8_GridSearch_CV/
 ┣ 📂 MiniProject_StockForecast/
 ┣ 📜 requirements.txt
 ┣ 📜 README.md
 ┗ 📜 Dockerfile
```

---

## 🏁 Conclusion
This repository represents the **complete practical implementation** of Machine Learning Build & Deployment techniques.  
It bridges the gap between **model creation, evaluation, and real-world deployment**, preparing students for professional ML workflows.

---

## 👨‍💻 Author
**Name:** Harish M  
**Register Number:** 2116-231501058  
**Course:** B.Tech AIML – 3rd Year  
**Institution:** Rajalakshmi Engineering College  
📧 harishm@email.com  
🌐 [GitHub Profile](https://github.com/harishm)
