# Lung Cancer Risk Prediction

A machine learning project focused on predicting lung cancer risk from demographic, behavioral, and clinical symptom data. This repository contains the complete workflow from exploratory data analysis (EDA) and feature engineering to model training, evaluation, and selection.

## 📖 Project Description

This project aims to develop a robust classification model to identify individuals at risk of lung cancer based on a set of 15 features, including smoking habits, symptoms like coughing and chest pain, and demographic information. The dataset, sourced from Kaggle, presents a significant class imbalance challenge, which is addressed using the ADASYN resampling technique.

Key aspects of the project include:
-   **Comprehensive EDA:** Detailed visual and statistical analysis to understand data distributions, correlations, and feature relationships.
-   **Class Imbalance Mitigation:** Application of ADASYN to generate synthetic samples for the minority class, improving model generalization.
-   **Extensive Model Comparison:** Evaluation of eight different machine learning algorithms, including Logistic Regression, Random Forest, XGBoost, and Support Vector Classifier.
-   **Feature Engineering:** Creation of interaction terms to capture complex relationships between symptoms and behaviors.

## 📊 Dataset Overview

The dataset contains information from 284 individuals (after cleaning duplicates) with 16 attributes.

**Target Variable:**
-   `LUNG_CANCER`: Binary classification (YES/NO)

**Feature Categories:**
-   **Demographic:** `GENDER`, `AGE`
-   **Behavioral:** `SMOKING`, `YELLOW_FINGERS`, `ALCOHOL CONSUMING`, `PEER_PRESSURE`
-   **Medical/Symptoms:** `ANXIETY`, `CHRONIC DISEASE`, `FATIGUE`, `ALLERGY`, `WHEEZING`, `COUGHING`, `SHORTNESS OF BREATH`, `SWALLOWING DIFFICULTY`, `CHEST PAIN`

## 🎯 Project Objectives

1.  **Prediction:** Build a machine learning model to accurately classify individuals as having a high risk of lung cancer or not.
2.  **Analysis:** Identify the most influential demographic, behavioral, and symptomatic factors contributing to lung cancer risk.
3.  **Evaluation:** Compare the performance of various classification algorithms to determine the most effective approach.
4.  **Imbalance Handling:** Address the severe class imbalance to ensure the model is effective at detecting the positive (cancer) class.

## 🛠️ Tech Stack

-   **Programming Language:** Python 3
-   **Libraries:**
    -   **Data Handling:** `pandas`, `numpy`
    -   **Visualization:** `matplotlib`, `seaborn`
    -   **Machine Learning:** `scikit-learn`, `imblearn` (for ADASYN), `xgboost`
-   **Environment:** Jupyter Notebook / Google Colab

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3 and `pip` installed on your system.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/lung-cancer-risk-prediction.git
    cd lung-cancer-risk-prediction
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  Download the dataset from [Kaggle](https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer/data) and place the `survey lung cancer.csv` file in the `data/` directory.
2.  Open and run the Jupyter Notebook `notebooks/lung_cancer_analysis.ipynb` to see the entire analysis, from EDA to model training.
    ```bash
    jupyter notebook notebooks/lung_cancer_analysis.ipynb
    ```

## 📈 Key Steps and Results

### 1. Data Preprocessing & Cleaning
-   Removed 33 duplicate entries.
-   Encoded categorical variables (e.g., 'M'/'F' -> 1/0, 'YES'/'NO' -> 1/0).
-   Standardized column names by stripping whitespace.

### 2. Exploratory Data Analysis (EDA)
-   Visualized the severe class imbalance (87% YES vs. 13% NO).
-   Identified age as a significant risk factor (median age ~65 for cancer cases).
-   Discovered strong gender-based patterns in smoking and alcohol consumption.
-   Analyzed correlations, finding features like `ALLERGY` and `SWALLOWING DIFFICULTY` to be most correlated with the target.

### 3. Feature Engineering
-   Created interaction terms: `CHEST PAIN_Alcohol`, `Wheezing_Coughing`, `ANXIETY_YELLOW_FINGERS`.

### 4. Handling Class Imbalance
-   Applied **ADASYN** to balance the training set, successfully creating a 1:1 class ratio.

### 5. Model Training & Evaluation
Eight models were trained and evaluated on a held-out test set. Key results:

| Model | Accuracy | AUC-ROC |
| :------------------------ | :------- | :------ |
| **Gaussian Naive Bayes**  | **94.6%**  | 91.7%   |
| **Support Vector Classifier** | 92.9%   | **97.5%**  |
| **Logistic Regression**   | 91.1%   | 97.2%   |
| Random Forest             | 91.1%   | 96.8%   |
| K-Nearest Neighbors       | 89.3%   | 92.6%   |

**Final Model Selection:** **Logistic Regression** was chosen as the best model due to its excellent combination of high accuracy (91.1%), high AUC (97.2%), and most importantly, its **consistency and top performance in cross-validation** (91.36% ± 2.23%), making it a reliable and interpretable choice.

## 🔮 Future Work

-   Experiment with advanced techniques like SMOTE and different undersampling strategies.
-   Perform hyperparameter tuning on the top-performing models (e.g., Logistic Regression, SVC) to further optimize performance.
-   Deploy the best model as a simple web application using Flask or Streamlit for interactive risk assessment.
-   Explore deep learning models for comparison.

## 👥 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/lung-cancer-risk-prediction/issues).

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

-   Dataset provided by [Nancy Al Aswad on Kaggle](https://www.kaggle.com/datasets/nancyalaswad90/lung-cancer).
-   Thanks to the open-source community for the invaluable Python libraries used in this project.

---

**Disclaimer:** This project is for educational and research purposes only. It is **not intended for actual medical diagnosis**. Always consult a healthcare professional for medical advice.
