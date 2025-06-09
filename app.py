from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Eğitilmiş model ve kullanılan sütunlar
pipeline, selected_columns = joblib.load("car_price_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Kullanıcıdan gelen veriler
    input_data = {
        "year": int(request.form["year"]),
        "make": request.form["make"],
        "body": request.form["body"],
        "transmission": request.form["transmission"],
        "condition": float(request.form["condition"]),
        "odometer": float(request.form["odometer"]),
        "color": request.form["color"],
        "interior": request.form["interior"],
        # Eksik sütunlar için varsayılan değerler:
        "model": "Base",        # örnek default model
        "trim": "Standard",     # örnek default trim
        "mmr": 15000.0          # örnek MMR değeri
    }

    # Modelin beklediği sıraya göre sütunları ayarla
    input_df = pd.DataFrame([input_data])[selected_columns]

    # Tahmin et
    predicted_price = pipeline.predict(input_df)[0]
    return render_template("index.html", prediction=round(predicted_price, 2))


if __name__ == '__main__':
    app.run(debug=True)
