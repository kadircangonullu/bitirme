# model.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb

from preprocessing import load_and_clean_data, preprocess_data, split_data

# Veriyi yükle ve işle
X, y = load_and_clean_data("car_prices.csv")

# Büyük veri seti için sadece ilk 40000 satırı al
X = X.head(40000)
y = y.head(40000)

preprocessor, cat_cols, num_cols = preprocess_data(X)
X_train, X_test, y_train, y_test = split_data(X, y)

# Modelleri tanımla (hafifletilmiş versiyonlar)
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=50, max_depth=8, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=50, max_depth=8, random_state=42),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor()
}

results = []

# Her model için pipeline kur, eğit, test et
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4)
    })

# Sonuçları göster
results_df = pd.DataFrame(results)
print(results_df)

# Grafik çiz
results_df.set_index("Model")[["MAE", "RMSE"]].plot(kind="bar", figsize=(10, 5), title="Model Performans Karşılaştırması")
plt.ylabel("Hata")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
