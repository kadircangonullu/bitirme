import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import joblib
import sqlite3
import os

app = Flask(__name__)
app.secret_key = 'secret-key'  # oturum için

# Model ve sütunlar
pipeline, model_columns = joblib.load("car_price_model.pkl")

# Veri seti örneği (benzer araçlar için)
df = pd.read_csv("car_prices.csv", on_bad_lines='skip')
df.columns = df.columns.str.strip().str.lower()
df = df[model_columns + ["sellingprice"]].dropna().head(50000)

# Kullanıcı veritabanı başlat
if not os.path.exists("users.db"):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)")
    # Yeni tablo: kullanıcı tahmin geçmişi
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


# Ana Sayfa
@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template("index.html")

# Kayıt Ol
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username = ?", (username,))
        if c.fetchone():
            flash("Kullanıcı adı zaten alınmış.")
            return redirect(url_for('register'))
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        flash("Kayıt başarılı! Giriş yapabilirsiniz.")
        return redirect(url_for('login'))
    return render_template("register.html")

# Giriş Yap
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
            flash("Hatalı giriş bilgisi.")
            return redirect(url_for('login'))
    return render_template("login.html")

# Çıkış Yap
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

# Tahmin Sayfası
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    predicted_price = None

    if request.method == 'POST':
        year = int(request.form["year"])
        make = request.form["make"]

        # Dinamik MMR tahmini
        filtered = df[(df["year"] == year) & (df["make"].str.lower() == make.lower())]
        if not filtered.empty:
            estimated_mmr = filtered["mmr"].mean()
        else:
            estimated_mmr = df["mmr"].mean()  # fallback

        # Tam model girdisi
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
        
        # Tahmini fiyatı session’a koyarsan benzer araçlar için kolay olur
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
            data_dict = json.loads(row[1])  # ← Burada str -> dict
        except json.JSONDecodeError:
            data_dict = {}
        predictions.append({
            "price": float(row[0]),
            "data": data_dict,
            "timestamp": row[2]
        })

    return render_template("account.html", username=session['username'], predictions=predictions)


@app.route('/similar')
def similar():
    if 'username' not in session or 'predicted_price' not in session:
        return redirect(url_for('login'))

    predicted_price = session['predicted_price']
    lower = predicted_price * 0.9
    upper = predicted_price * 1.1

    similar = df[(df['sellingprice'] >= lower) & (df['sellingprice'] <= upper)].head(10)

    return render_template("similar_cars.html",
                           prediction=predicted_price,
                           cars=similar.to_dict(orient="records"))


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
        flash("Şifreniz başarıyla güncellendi.")
    else:
        conn.close()
        flash("Eski şifre hatalı. Lütfen tekrar deneyin.")

    return redirect(url_for('account'))



if __name__ == '__main__':
    app.run(debug=True)
