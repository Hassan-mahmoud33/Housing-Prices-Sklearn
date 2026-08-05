import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import os
import re

IN_KAGGLE = os.path.exists('/kaggle/input')

if IN_KAGGLE:
    PROJECT_ROOT = Path("/kaggle/working")
    DATA_DIR = Path('/kaggle/input/house-prices-advanced-regression-techniques')
    MODEL_PATH = PROJECT_ROOT / "Cat_Boost_Regressor_model.pkl"
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "datasets"
    MODEL_PATH = PROJECT_ROOT / "models" / "Cat_Boost_Regressor_model.pkl"

# قراءة بيانات التست المعالجة
test_data = pd.read_csv(DATA_DIR / "processed" / "test_processed.csv")

# الاحتفاظ بالـ Id
test_ids = test_data["Id"]

# حذف الأعمدة التي تم حذفها أثناء التدريب
test_data = test_data.drop(
    columns=["Id", "MoSold", "YrSold"],
    errors="ignore"
)

# تنظيف أسماء الأعمدة
test_data.columns = [
    re.sub(r'[^A-Za-z0-9_]+', '_', col)
    for col in test_data.columns
]

# تحميل أفضل موديل
model = joblib.load(MODEL_PATH)

# التنبؤ
pred = model.predict(test_data)

# الرجوع من Log إلى السعر الحقيقي
pred = np.expm1(pred)

# إنشاء ملف Submission
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": pred
})

# حفظ الملف
submission.to_csv(
    PROJECT_ROOT / "submission.csv",
    index=False
)

print(submission.head())
print("\n✅ submission.csv saved successfully")
print(f"Saved to: {PROJECT_ROOT / 'submission.csv'}")