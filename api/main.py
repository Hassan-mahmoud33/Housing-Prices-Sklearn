from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import numpy as np
import joblib as jb
import json
from pathlib import Path

# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'
FRONDEND_DIR = Path(__file__).resolve().parent.parent / 'front_end'

MODEL_PATH = MODELS_DIR / 'Cat_Boost_Regressor_model.pkl'
FEATURE_CONFIG_PATH = MODELS_DIR / 'feature_configration.json'

# ---------------------------------------------------------------------------

model = jb.load(MODEL_PATH)

with open(FEATURE_CONFIG_PATH, 'r', encoding='utf-8') as f:
    FEATURE_CONFIG = json.load(f)

MODEL_COLUMNS = FEATURE_CONFIG['model_columns']
# All 232 columns used during training, in the same order

NUMERIC_DEFAULTS = FEATURE_CONFIG['numeric_defaults']
# Median value for numeric features that the user does not enter

CATEGORICAL_DEFAULTS = FEATURE_CONFIG['categorical_defaults']
# Most common value for categorical features that the user does not enter

CATEGORICAL_GROUPS = FEATURE_CONFIG['categorical_groups']
# All one-hot encoded categorical groups and their columns

CATEGORICAL_OPTIONS = FEATURE_CONFIG['categorical_options']
# Available options for each categorical feature

NEIGHBORHOOD_OPTIONS = tuple(CATEGORICAL_OPTIONS['Neighborhood'])
KITCHEN_QUAL_OPTIONS = tuple(CATEGORICAL_OPTIONS['KitchenQual'])

# ---------------------------------------------------------------------------

app = FastAPI(title='< House Price Prediction API >')

app.add_middleware(
    CORSMiddleware,
    allow_headers=["*"]
)

# ---------------------------------------------------------------------------

# The user enters only these 15 important features.
# The remaining features are filled inside build_features().
class HouseData(BaseModel):
    OverallQual: int                       # Overall material and finish quality (1 - 10)
    OverallCond: int                       # Overall condition of the house (1 - 10)
    GrLivArea: float                       # Above-ground living area (sq ft)
    TotalBsmtSF: float                     # Total basement area (sq ft)
    FirstFlrSF: float                      # First floor area (sq ft)
    SecondFlrSF: float                     # Second floor area (sq ft), 0 if there is no second floor
    TotalBathrooms: float                  # Total number of bathrooms (half bath = 0.5)
    BsmtFinSF1: float                       # Finished basement area (sq ft)
    LotArea: float                          # Total lot area (sq ft)
    GarageCars: float                       # Garage capacity in cars
    GarageArea: float                       # Garage area (sq ft)
    YearBuilt: int                          # Year the house was built
    YearRemodAdd: int                       # Year of the last remodeling, same as YearBuilt if never remodeled
    Neighborhood: Literal[NEIGHBORHOOD_OPTIONS]  # Neighborhood
    KitchenQual: Literal[KITCHEN_QUAL_OPTIONS]   # Kitchen quality


# ======================================================================================================

def build_features(data: HouseData) -> pd.DataFrame:

    row = {}

    # 1) Fill numeric features that the user does not enter
    #    with their median values from the training data.
    row.update(NUMERIC_DEFAULTS)

    # 2) Fill categorical features with 0 first.
    #    Then set the most common category to 1.
    for group, columns in CATEGORICAL_GROUPS.items():
        for col in columns:
            row[col] = 0

        if group in CATEGORICAL_DEFAULTS:
            default_col = f"{group}_{CATEGORICAL_DEFAULTS[group]}"
            row[default_col] = 1

    # 3) Add the numeric features entered by the user.
    row['OverallQual'] = data.OverallQual
    row['OverallCond'] = data.OverallCond
    row['GrLivArea'] = data.GrLivArea
    row['TotalBsmtSF'] = data.TotalBsmtSF
    row['1stFlrSF'] = data.FirstFlrSF
    row['2ndFlrSF'] = data.SecondFlrSF
    row['TotalBathrooms'] = data.TotalBathrooms
    row['BsmtFinSF1'] = data.BsmtFinSF1
    row['LotArea'] = data.LotArea
    row['GarageCars'] = data.GarageCars
    row['GarageArea'] = data.GarageArea
    row['YearBuilt'] = data.YearBuilt
    row['YearRemodAdd'] = data.YearRemodAdd

    # 4) Add the categorical features selected by the user.
    #    Set the selected neighborhood and kitchen quality to 1.
    row[f'Neighborhood_{data.Neighborhood}'] = 1
    row[f'KitchenQual_{data.KitchenQual}'] = 1

    # 5) Apply the same feature engineering used during model training.
    sale_year = NUMERIC_DEFAULTS['YrSold']
    # Use the median sale year because the user does not enter a sale year.

    row['TotalSF'] = (
        row['TotalBsmtSF']
        + row['1stFlrSF']
        + row['2ndFlrSF']
    )

    row['HouseAge'] = sale_year - row['YearBuilt']
    row['YearsSinceRemodel'] = sale_year - row['YearRemodAdd']
    row['QualityArea'] = row['OverallQual'] * row['GrLivArea']

    row['HasGarage'] = int(row['GarageArea'] > 0)
    row['HasBasement'] = int(row['TotalBsmtSF'] > 0)

    df = pd.DataFrame([row])

    try:
        # Keep only the model features and use the same order as training.
        df = df[MODEL_COLUMNS]

    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Missing feature column: {e}"
        )

    return df


# ======================================================================================================


@app.post('/predict')
def predict(data: HouseData):

    features = build_features(data)

    log_pred = model.predict(features)[0]

    # The model was trained using np.log1p(SalePrice).
    # Convert the prediction back to the original dollar scale.
    predicted_price = float(np.expm1(log_pred))

    return {
        'predicted_price': round(predicted_price, 2)
    }


# ----------------------------------------------------------------------------------

# Serve the frontend files.
app.mount(
    "/",
    StaticFiles(directory=FRONDEND_DIR, html=True),
    name='frond_end'
)
