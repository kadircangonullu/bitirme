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

def group_feature_importances(feature_names, importances):
    grouped = {}
    for name, importance in zip(feature_names, importances):
        if name.startswith("cat__"):
            group = name.split("__")[1].split("_")[0]  # ex: cat__make_Toyota -> make
        elif name.startswith("num__"):
            group = name.replace("num__", "")          # num__odometer -> odometer
        else:
            group = name
        grouped[group] = grouped.get(group, 0) + importance
    df = pd.DataFrame(list(grouped.items()), columns=["Feature", "Importance"])
    return df.sort_values("Importance", ascending=False)

def plot_grouped_importance(grouped_df, model_name):
    grouped_df["Importance"] = grouped_df["Importance"] / grouped_df["Importance"].sum()
    top = grouped_df.head(5)  # Get top 5 features
    plt.figure(figsize=(8, 5))
    plt.barh(top["Feature"], top["Importance"])
    plt.title(f"{model_name} - Grouped Feature Importance")
    plt.xlabel("Normalized Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def main():
    X, y = load_and_clean_data("car_prices.csv")
    X = X.head(40000)
    y = y.head(40000)

    preprocessor, cat_cols, num_cols = preprocess_data(X)
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=50, max_depth=8, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=50, max_depth=8, random_state=42),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor()
    }

    results = []

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

        # Özellik önemini çiz
        if hasattr(model, "feature_importances_"):
            feature_names = preprocessor.get_feature_names_out()
            importances = model.feature_importances_
            grouped_df = group_feature_importances(feature_names, importances)
            plot_grouped_importance(grouped_df, name)

    # Tüm model skorlarını çiz
    results_df = pd.DataFrame(results)
    print(results_df)

    results_df.set_index("Model")[["MAE", "RMSE"]].plot(kind="bar", figsize=(10, 5), title="Model Performance Comparison")
    plt.ylabel("Error")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    results_df.set_index("Model")["R2 Score"].plot(kind="barh", color="green", title="R2 Score Comparison")
    plt.xlabel("R2 Score")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
