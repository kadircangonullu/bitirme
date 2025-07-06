import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import joblib
import sqlite3
import os
import requests

app = Flask(__name__)
app.secret_key = 'secret-key'  # required for session management

# Model and columns
pipeline, model_columns = joblib.load("car_price_model.pkl")

# Sample dataset (for similar cars)
df = pd.read_csv("car_prices.csv", on_bad_lines='skip')
df.columns = df.columns.str.strip().str.lower()
df = df[model_columns + ["sellingprice"]].dropna().head(50000)

# Initialize user database
if not os.path.exists("users.db"):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)")
    # New table: user prediction history
    c.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY,
            username TEXT,
            price REAL,
            data TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


# Home Page
@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template("index.html")

# Register Page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        if c.fetchone():
            flash("Username Already Taken.")
            return redirect(url_for('register'))
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        flash("Registration successful! You can log in.")
        return redirect(url_for('login'))
    return render_template("register.html")

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
        user = c.fetchone()
        conn.close()
        if user:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash("Invalid login credentials.")
            return redirect(url_for('login'))
    return render_template("login.html")

# Logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

# Prediction Page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    predicted_price = None

    if request.method == 'POST':
        year = int(request.form["year"])
        make = request.form["make"]

        # Dynamic MMR estimation
        filtered = df[(df["year"] == year) & (df["make"].str.lower() == make.lower())]
        if not filtered.empty:
            estimated_mmr = filtered["mmr"].mean()
        else:
            estimated_mmr = df["mmr"].mean()  # fallback

        # Final model input
        data = {
            "year": year,
            "make": make,
            "body": request.form["body"],
            "transmission": request.form["transmission"],
            "condition": float(request.form["condition"]),
            "odometer": float(request.form["odometer"]),
            "color": request.form["color"],
            "interior": request.form["interior"],
            "model": "Base",
            "trim": "Standard",
            "mmr": estimated_mmr
        }

        input_df = pd.DataFrame([data])[model_columns]
        predicted_price = round(pipeline.predict(input_df)[0], 2)

        # Store predicted price in session for similar cars
        session['predicted_price'] = float(predicted_price)
        
        record_data = json.dumps(data)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        price = float(predicted_price)

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("""INSERT INTO predictions (username, price, data, timestamp)
                    VALUES (?, ?, ?, ?)""", (session['username'], price, record_data, timestamp))
        conn.commit()
        conn.close()

    return render_template("predict.html", predicted_price=predicted_price)

@app.route('/account')
def account():
    if 'username' not in session:
        return redirect(url_for('login'))

    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT price, data, timestamp FROM predictions WHERE username = ?", (session['username'],))
    rows = c.fetchall()
    conn.close()

    predictions = []
    for row in rows:
        try:
            data_dict = json.loads(row[1])
            price = row[0]

            # Filter similar cars
            # Assuming 'sellingprice' is the target column in df
            lower = price * 0.9
            upper = price * 1.1
            similar_cars = df[(df['sellingprice'] >= lower) & (df['sellingprice'] <= upper)].head(3)

            predictions.append({
                "price": price,
                "data": data_dict,
                "timestamp": row[2],
                "similars": similar_cars.to_dict(orient="records")
            })
        except Exception as e:
            print("Skipped Missing Value:", e)

    return render_template("account.html", username=session['username'], predictions=predictions)


def fetch_car_image(car_name):
    api_key = "042fe547412ec396f00e07c5fb3c1cdafc67804d6fd8bb4704d4d5e7cd1f0a12"
    params = {
        "q": car_name + " car",
        "tbm": "isch",
        "ijn": "0",
        "api_key": api_key
    }

    response = requests.get("https://serpapi.com/search.json", params=params)
    data = response.json()

    try:
        first_image_url = data['images_results'][0]['original']
        return first_image_url
    except (KeyError, IndexError):
        return "/static/default_car.jpg"  # fallback image

@app.route('/similar')
def similar():
    if 'username' not in session or 'predicted_price' not in session:
        return redirect(url_for('login'))

    predicted_price = session['predicted_price']
    lower = predicted_price * 0.9
    upper = predicted_price * 1.1

    similar = df[(df['sellingprice'] >= lower) & (df['sellingprice'] <= upper)].head(10)
    
    cars = similar.to_dict(orient="records")
    for car in cars:
        name = f"{car['year']} {car['make']} {car['model']}"
        car['image_url'] = fetch_car_image(name)

    return render_template("similar_cars.html",
                           prediction=predicted_price,
                           cars=cars)

@app.route('/api/similar_for_price/<float:price>')
def similar_for_price(price):
    if 'username' not in session:
        return {"error": "Giriş yapılmamış."}, 401

    lower = price * 0.9
    upper = price * 1.1

    similar = df[(df['sellingprice'] >= lower) & (df['sellingprice'] <= upper)].head(10)

    results = similar[["year", "make", "model", "sellingprice", "body", "transmission", "odometer", "color", "interior"]]
    return results.to_dict(orient="records")


@app.route('/change_password', methods=['POST'])
def change_password():
    if 'username' not in session:
        return redirect(url_for('login'))

    old_password = request.form['old_password']
    new_password = request.form['new_password']
    username = session['username']

    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = c.fetchone()

    if row and row[0] == old_password:
        c.execute("UPDATE users SET password = ? WHERE username = ?", (new_password, username))
        conn.commit()
        conn.close()
        flash("Your password has been successfully updated.")
    else:
        conn.close()
        flash("Old password is incorrect. Please try again.")

    return redirect(url_for('account'))



if __name__ == '__main__':
    app.run(debug=True)
