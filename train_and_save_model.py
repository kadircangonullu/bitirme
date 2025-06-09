# train_and_save_model.py

import pandas as pd
import joblib
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing import load_and_clean_data, preprocess_data, split_data

# Veriyi yükle
df = pd.read_csv("car_prices.csv", on_bad_lines='skip')
df.columns = df.columns.str.strip().str.lower()  # sütunları temizle

# Gerekli sütunlar
selected_columns = ["year", "make", "model", "trim", "body", "transmission", "condition",
                    "odometer", "color", "interior", "mmr", "sellingprice"]

# Hata kontrolü
for col in selected_columns:
    if col not in df.columns:
        raise ValueError(f"Sütun eksik: {col}")

# Veri seçimi ve temizlik
df = df[selected_columns].dropna().head(40000)
X = df.drop(columns=["sellingprice"])
y = df["sellingprice"]

# Ön işleme ve veri bölme
preprocessor, _, _ = preprocess_data(X)
X_train, X_test, y_train, y_test = split_data(X, y)

# Model tanımı
model = xgb.XGBRegressor(n_estimators=50, max_depth=8, random_state=42)

# Pipeline oluştur
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", model)
])

# Eğit
pipeline.fit(X_train, y_train)

# Performans değerlendirme
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("Model Performansı:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2 Score: {r2:.4f}")

# Kaydet
joblib.dump((pipeline, X.columns.tolist()), "car_price_model.pkl")
print("Model başarıyla kaydedildi: car_price_model.pkl")
