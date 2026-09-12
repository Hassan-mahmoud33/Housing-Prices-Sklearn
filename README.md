# 🏠 House Prices - Advanced Regression Techniques

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-3.0.5-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.5.1-013243?logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-red)
![LightGBM](https://img.shields.io/badge/LightGBM-Regression-green)
![CatBoost](https://img.shields.io/badge/CatBoost-Regression-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A complete machine learning pipeline for predicting house sale prices, built on the classic Kaggle **House Prices - Advanced Regression Techniques** competition dataset. The project covers the full workflow from raw data to a submission-ready model: cleaning, outlier handling, feature engineering, model training, evaluation, and final predictions.

📊 **Competition result:** ranked **1274 / 3500** on the public leaderboard.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Workflow](#workflow)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Overview

The goal of this project is to predict the final sale price of residential homes in Ames, Iowa, based on 79 explanatory variables describing almost every aspect of the property (lot size, quality, condition, basement, garage, and more).

The notebook works in stages, each saved as a separate file, so every step of the pipeline is easy to follow, re-run, and debug on its own.

## Project Structure

```
housing-prices/
│
├── datasets/
│   ├── raw/                          # Original competition files
│   │   ├── data.csv
│   │   ├── sample_submission.csv
│   │   ├── test.csv
│   │   └── train.csv
│   │
│   └── processed/                    # Cleaned & feature-engineered data
│       ├── cleaned_train_data.csv
│       ├── cleaned_test_data.csv
│       ├── final_train.csv
│       ├── final_test.csv
│       ├── model_evaluation_results.csv
│       └── submission.csv
│
├── models/                           # Trained models (saved with joblib)
│   ├── Linear_Regression_model.pkl
│   ├── ridge_model.pkl
│   ├── Random_Forest_Regressor_model.pkl
│   ├── Gradient_Boosting_Regressor_model.pkl
│   ├── XGB_Regressor_model.pkl
│   ├── LGBM_Regressor_model.pkl
│   └── Cat_Boost_Regressor_model.pkl
│
├── notebooks/
│   ├── 01_EDA_cleaning_train.ipynb   # EDA + cleaning for the training set
│   ├── 01.2_claning_Test.ipynb       # Cleaning for the test set
│   ├── 02_feature_engineering.ipynb  # New features + encoding + selection
│   ├── 03_train_model.ipynb          # Training multiple regression models
│   ├── 04_Evaluate_models.ipynb      # Comparing models & picking the best one
│   ├── 05_predictions.ipynb          # Generating the final submission file
│   └── catboost_info/                # Auto-generated CatBoost training logs
│
└── requirements.txt
```

## Workflow

The pipeline is split into five main notebooks, run in this order:

1. **`01_EDA_cleaning_train.ipynb`** – Exploratory data analysis on the training set, handling missing values (both by filling and by dropping high-null columns like `Alley`, `Fence`, `MiscFeature`, `MasVnrType`, `FireplaceQu`), detecting and removing outliers (e.g. abnormal `LotFrontage` and `BsmtFinSF1` values), and applying a log transformation (`log1p`) to `SalePrice` to reduce skewness.
2. **`01.2_claning_Test.ipynb`** – The same cleaning logic applied to the test set, using statistics (mode, median) learned from the training data to avoid data leakage.
3. **`02_feature_engineering.ipynb`** – Creating new, more informative features from the existing ones, one-hot encoding categorical columns, and selecting the most relevant features with `SelectPercentile`.
4. **`03_train_model.ipynb`** – Training several regression models on the processed data.
5. **`04_Evaluate_models.ipynb`** – Comparing model performance and choosing the best one.
6. **`05_predictions.ipynb`** – Loading the best model and generating the final `submission.csv` file for Kaggle.

## Feature Engineering

A set of new features was engineered to capture patterns that aren't obvious from the raw columns alone:

| Feature | Description |
|---|---|
| `HouseAge` | Age of the house at the time of sale |
| `TotalSF` | Total square footage (basement + 1st floor + 2nd floor) |
| `TotalBathrooms` | Combined full and half bathrooms (basement included) |
| `TotalPorchSF` | Total porch/deck area combined |
| `NonBedroomRooms` | Rooms above ground excluding bedrooms |
| `HasGarage` | Whether the house has a garage (1/0) |
| `HasBasement` | Whether the house has a basement (1/0) |
| `HasFirePlace` | Whether the house has a fireplace (1/0) |
| `HasPool` | Whether the house has a pool (1/0) |
| `YearsSinceRemodel` | Years passed since the last remodel |
| `GarageAreaPerCar` | Average garage area per car |
| `GarageAge` | Age of the garage at the time of sale |
| `QualityArea` | Overall quality multiplied by living area |
| `TotalSF_per_Room` | Total square footage divided by number of rooms |

After checking correlation between features, `HasPool` and `HasFirePlace` were dropped due to high correlation with other existing columns.

## Models

The following regression models were trained and compared:

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGB Regressor
- LGBM Regressor
- CatBoost Regressor

📈 **Best performing model:** `XGBRFRegressor`, which was used to generate the final predictions.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/<your-username>/housing-prices.git
cd housing-prices
pip install -r requirements.txt
```

**requirements.txt**
```
pandas==3.0.5
numpy==2.5.1
matplotlib==3.11.1
seaborn
scikit-learn
xgboost
lightgbm
catboost
joblib
```

## How to Run

Run the notebooks in order from the `notebooks/` folder:

```
01_EDA_cleaning_train.ipynb   →   01.2_claning_Test.ipynb   →   02_feature_engineering.ipynb   →   03_train_model.ipynb   →   04_Evaluate_models.ipynb   →   05_predictions.ipynb
```

Each notebook reads its input from `datasets/raw/` or `datasets/processed/` (depending on the stage) and writes its output back to `datasets/processed/`. The final submission file will be generated at `datasets/processed/submission.csv`, ready to upload to Kaggle.

> **Note:** The code automatically detects whether it's running inside a Kaggle notebook (`/kaggle/input`) or locally, and adjusts the data paths accordingly — no manual path changes needed.

## Results

- Leaderboard position: **1274 out of 3500** participants.
- Best model: **XGBRFRegressor**.
- Full comparison of all trained models is available in `datasets/processed/model_evaluation_results.csv`.

## Tech Stack

- **Language:** Python
- **Data handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, XGBoost, LightGBM, CatBoost
- **Model persistence:** Joblib

## License

This project is licensed under the MIT License.
