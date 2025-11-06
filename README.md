# 🧠 Introverts vs Extroverts Classification

This project builds a machine learning model to distinguish between Introvert and Extrovert personality types using survey-based social behavior features.

---

## 📂 Project Structure
```
project/
│
├── data/
│   ├── raw/                     # Original dataset from Kaggle
│   │   ├── train.csv
│   │   └── test.csv
│   └── preprocessed/            # Created automatically after preprocessing
│       ├── train_df.csv
│       └── test_df.csv
│
├── submissions/                 # Generated model predictions
│ 
│
├── src/
│   ├── data_preprocessing.py    # Data cleaning, feature engineering, encoding
│   └── model_training.py         # Model training, validation, performance evaluation
│
├── notebooks/                   # Optional exploratory analysis and experiments
│   ├── EDA.ipynb
│   └── models.ipynb
│
├── main.py                      # Runs full pipeline (preprocess + train + submit)
├── requirements.txt
└── README.md
```

---

## 📥 Dataset
Download the dataset from Kaggle:
https://www.kaggle.com/datasets/nehalbirla/introvert-extrovert-classification

Place the downloaded files here:
```
data/raw/train.csv
data/raw/test.csv
```

---

## ⚙️ Setup & Installation
```bash
git clone <your-repo-url>
cd Introverts-from-Extroverts


# (Optional) Create virtual environment
python3 -m venv venv
source venv/bin/activate     # macOS / Linux
venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🔄 Data Preprocessing
The preprocessing pipeline performs:

1. Dataset overview and missing value check
2. Visualization of missing features
3. Creation of a custom feature `sum` based on user responses
4. Smart row-wise value imputation using behavioral patterns
5. Converting feature types for efficiency
6. Label encoding binary and target columns

Run manually:
```bash
python src/data_preprocessing.py
```

---

## 🤖 Model Training
The project currently trains two models for comparison:
- **Random Forest Classifier**
- **LightGBM Classifier**
- **SVM
- **XGBoost

Each model is evaluated using:
- Accuracy
- F1 Score (macro)
- Confusion Matrix

Run training manually:
```bash
python src/model_training.py
```

---

## 🚀 Full Pipeline Execution
To run preprocessing **and** training together:
```bash
python main.py
```
This will:
- Clean and encode the dataset
- Train the models
- Create prediction submission files

Output location:
```
submissions/RandomForest.csv
```

---

## 🧠 Output Interpretation
The model predicts binary classes:
```
0 → Extrovert
1 → Introvert
```

---

## ✅ Requirements
Install everything with:
```bash
pip install -r requirements.txt
```

---

## 🎯 Summary
This project provides a reproducible ML pipeline that:
- Processes behavioral survey data
- Learns personality type patterns
- Outputs predictions ready for Kaggle submission

Feel free to improve feature engineering, tune models, or add new classifiers!

