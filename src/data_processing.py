"""
data_processing.py
--------------------
دوال معالجة بيانات قابلة لإعادة الاستخدام على الترين والتست بنفس الطريقة.

الفكرة الأساسية:
- أي "إحصائية" بتتحسب (median, mode, أعمدة الـ dummies) لازم تتحسب من الترين بس،
  وتتحفظ، وبعدين تتطبق على التست.
- الـ Outliers بتتشال من الترين بس، أبداً من التست.
- Log transform للـ SalePrice بيتطبق على الترين بس (لأن التست مفيهوش target أصلاً).
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1) إزالة الـ Outliers (يُستخدم على الترين فقط)
# ---------------------------------------------------------------------------
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """يشيل الصفوف الشاذة بالـ index بتاعها بالظبط (زي ما اتحددت في الـ EDA). للترين فقط."""
    df = df.copy()

    df = df.drop(index=1298, errors='ignore')  # LotFrontage outlier
    df = df.drop(index=523, errors='ignore')   # BsmtFinSF1 outlier

    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2) تعبئة القيم الفارغة الفئوية (Categorical) - نفس المنطق يتكرر في التست
# ---------------------------------------------------------------------------
def fill_categorical_nulls(df: pd.DataFrame, train_mode_electrical=None) -> pd.DataFrame:
    """
    يعبي كل الأعمدة الفئوية الفاضية بـ 'None'، ما عدا Electrical
    اللي بتتعبى بالـ mode بتاع الترين.

    train_mode_electrical: القيمة المحسوبة من الترين (مررها وقت التست).
                            لو None، هيتحسب من نفس الداتا (استخدمها وقت الترين بس).
    """
    df = df.copy()

    cat_cols = df.select_dtypes(include=['object', 'string']).columns
    cat_cols = cat_cols.drop('Electrical', errors='ignore')

    df[cat_cols] = df[cat_cols].fillna('None')

    if train_mode_electrical is None:
        train_mode_electrical = df['Electrical'].mode()[0]

    df['Electrical'] = df['Electrical'].fillna(train_mode_electrical)

    return df, train_mode_electrical


# ---------------------------------------------------------------------------
# 3) تعبئة القيم الفارغة الرقمية (Numerical) - باستخدام median الترين فقط
# ---------------------------------------------------------------------------
def fill_numerical_nulls(df: pd.DataFrame, train_medians: pd.Series = None):
    """
    يعبي GarageYrBlt بـ 0، وباقي الأعمدة الرقمية بالـ median.

    train_medians: الـ Series المحسوبة من الترين (مررها وقت التست).
                    لو None، هتتحسب من نفس الداتا (استخدمها وقت الترين بس).
    """
    df = df.copy()

    if 'GarageYrBlt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)

    num_cols = df.select_dtypes(include=np.number).columns
    num_cols = num_cols.drop('GarageYrBlt', errors='ignore')
    # لو 'SalePrice' موجود (في الترين بس) منستخدموش نفسه في التعبئة
    num_cols = num_cols.drop('SalePrice', errors='ignore')

    null_num_cols = df[num_cols].columns[df[num_cols].isnull().any()]

    if train_medians is None:
        train_medians = df[null_num_cols].median()

    df[null_num_cols] = df[null_num_cols].fillna(train_medians)

    return df, train_medians


# ---------------------------------------------------------------------------
# 4) Feature Engineering - نفسه بالظبط على الترين والتست
# ---------------------------------------------------------------------------
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    df['TotalBathrooms'] = (
        df['FullBath'] + (0.5 * df['HalfBath']) +
        df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])
    )

    df['TotalPorchSF'] = (
        df['OpenPorchSF'] + df['3SsnPorch'] +
        df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
    )

    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']

    df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
    df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['Has2ndFlr'] = (df['2ndFlrSF'] > 0).astype(int)

    return df


# ---------------------------------------------------------------------------
# 5) Encoding - لازم أعمدة التست تطابق أعمدة الترين بالظبط
# ---------------------------------------------------------------------------
def encode_categorical(df: pd.DataFrame, train_columns=None):
    """
    يعمل get_dummies، ولو train_columns اتبعتت (وقت التست)،
    بيعمل reindex عشان الأعمدة تبقى مطابقة تماماً لأعمدة الترين
    (أي عمود ناقص هيتحط صفر، وأي عمود زيادة هيتشال).
    """
    df = pd.get_dummies(df, drop_first=True)
    df = df.astype(int, errors='ignore')

    if train_columns is not None:
        df = df.reindex(columns=train_columns, fill_value=0)

    return df


# ---------------------------------------------------------------------------
# 6) Pipeline كامل للترين - بيرجع الداتا المعالجة + كل الإحصائيات المحفوظة
# ---------------------------------------------------------------------------
def process_train(train_data: pd.DataFrame):
    train_data = train_data.copy()

    train_data = remove_outliers(train_data)

    train_data, electrical_mode = fill_categorical_nulls(train_data)
    train_data, num_medians = fill_numerical_nulls(train_data)

    # Log transform لـ SalePrice (الترين بس)
    train_data['SalePrice'] = np.log1p(train_data['SalePrice'])

    train_data = add_engineered_features(train_data)
    train_data = encode_categorical(train_data)

    stats = {
        'electrical_mode': electrical_mode,
        'num_medians': num_medians,
        'train_columns': train_data.columns,
    }

    return train_data, stats


# ---------------------------------------------------------------------------
# 7) Pipeline كامل للتست - بياخد نفس الإحصائيات اللي طلعت من الترين
# ---------------------------------------------------------------------------
def process_test(test_data: pd.DataFrame, stats: dict):
    test_data = test_data.copy()

    # ملاحظة: مفيش remove_outliers هنا نهائياً

    test_data, _ = fill_categorical_nulls(
        test_data, train_mode_electrical=stats['electrical_mode']
    )
    test_data, _ = fill_numerical_nulls(
        test_data, train_medians=stats['num_medians']
    )

    # مفيش log transform هنا لأن مفيش SalePrice في التست

    test_data = add_engineered_features(test_data)

    # لازم نشيل SalePrice من أعمدة الترين وقت المطابقة لأنه مش موجود في التست
    train_cols_no_target = stats['train_columns'].drop('SalePrice', errors='ignore')
    test_data = encode_categorical(test_data, train_columns=train_cols_no_target)

    return test_data


# ---------------------------------------------------------------------------
# التشغيل الفعلي: بيقرا train.csv و test.csv من datasets/raw
# ويحفظ train_processed.csv و test_processed.csv في datasets/processed
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # هذا الملف موجود في housing-prices/src/data_processing.py
    # فـ parent.parent بيوصلنا لـ housing-prices/ (جذر المشروع)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    RAW_DIR = PROJECT_ROOT / 'datasets' / 'raw'
    PROCESSED_DIR = PROJECT_ROOT / 'datasets' / 'processed'
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Reading raw data from: {RAW_DIR}')
    train_data_raw = pd.read_csv(RAW_DIR / 'train.csv')
    test_data_raw = pd.read_csv(RAW_DIR / 'test.csv')

    print(f'Train raw shape: {train_data_raw.shape}')
    print(f'Test raw shape:  {test_data_raw.shape}')

    train_processed, stats = process_train(train_data_raw)
    test_processed = process_test(test_data_raw, stats)

    print(f'Train processed shape: {train_processed.shape}')
    print(f'Test processed shape:  {test_processed.shape}')

    train_processed.to_csv(PROCESSED_DIR / 'train_processed.csv', index=False)
    test_processed.to_csv(PROCESSED_DIR / 'test_processed.csv', index=False)

    print(f'Saved: {PROCESSED_DIR / "train_processed.csv"}')
    print(f'Saved: {PROCESSED_DIR / "test_processed.csv"}')