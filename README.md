
# 🚗 The Cars – AI-Powered Used Car Price Estimation System

**The Cars** is a machine learning-based web application designed to predict the fair market value of second-hand cars. Built with a robust ML pipeline and an interactive front-end, the platform enables users to receive instant price estimates and discover similar vehicles based on selected parameters.

---

## 🧠 Project Overview

This project leverages a comprehensive dataset of vehicle listings and applies multiple regression algorithms—including XGBoost, Random Forest, and Gradient Boosting—to predict car prices based on features such as year, brand, model, mileage, condition, and more. It also offers:

- A user-friendly web interface built with Flask and HTML/CSS
- A detailed model performance comparison and visualizations
- A similar car recommendation system
- Language toggle support (Turkish/English)
- Dark mode and responsive design
- Image-enhanced output using Google Images API

---

## 📊 Features

- **Accurate Price Prediction** using trained regression models
- **User Login & History Tracking** of past predictions
- **Recommended Cars** based on predicted price range
- **Interactive Visualization** of model performance and feature importance
- **Fully Responsive Design** with clean UI/UX
- **Multilingual Support (TR/EN)**
- **Automated Car Image Retrieval** from Google

---

## 🛠️ Tech Stack

| Category         | Tools/Frameworks                              |
|------------------|-----------------------------------------------|
| Programming      | Python, HTML5, CSS3, JavaScript               |
| Backend          | Flask                                          |
| Machine Learning | Scikit-learn, XGBoost, Pandas, NumPy          |
| Frontend         | Bootstrap (Custom), Font Awesome              |
| Visualization    | Matplotlib, Seaborn                           |
| Database         | CSV (for model training), SQLite (optional)  |
| Others           | Google Custom Search API (for images)        |

---

## 📁 Project Structure

```
The-Cars/
│
├── static/               # CSS, images, JS files
├── templates/            # HTML templates
├── model.py              # ML model training & evaluation
├── preprocessing.py      # Data cleaning and preprocessing functions
├── app.py                # Main Flask application
├── car_prices.csv        # Raw dataset
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## 📈 Model Performance

After training and testing six different regression models, the results showed that **XGBoost** provided the best accuracy, followed closely by **Gradient Boosting** and **Random Forest**. Feature importance analysis confirmed that **year**, **odometer**, and **condition** are the most influential features.

| Model             | MAE    | RMSE   | R² Score |
|------------------|--------|--------|----------|
| Linear Regression| 3693.70| 5015.45| 0.7299   |
| Random Forest    | 3216.28| 3864.12| 0.8397   |
| XGBoost          | 1945.42| 3213.21| 0.8583   |
| SVR              | 6145.24| 9099.79| 0.1109   |
| KNN              | 1975.28| 3633.33| 0.8535   |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/the-cars.git
cd the-cars
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Flask application:

```bash
python app.py
```

Then navigate to `http://127.0.0.1:5000` in your browser.

---

## 📂 Dataset

The dataset includes over 100,000 vehicle records with features such as:

- `year`: Year of manufacture  
- `make`: Brand (e.g., Toyota, BMW)  
- `model`: Specific car model  
- `trim`: Submodel details  
- `body`: Body type (e.g., sedan, SUV)  
- `transmission`: Automatic or manual  
- `state`: Registration state  
- `condition`: Overall vehicle condition  
- `odometer`: Mileage  
- `color`, `interior`, `mmr`, `sellingprice`: Various vehicle details and target variable

---

## 📌 Future Improvements

- Integration with real-time listings via API
- Deep learning models for price estimation
- Enhanced filtering for car recommendations
- Mobile app version using Flutter

---

## 🙏 Acknowledgements

Special thanks to the developers and contributors of open-source libraries such as Scikit-learn, Flask, and XGBoost. Also, credit to the [OpenAI ChatGPT](https://openai.com/chatgpt) platform for assistance in project ideation and documentation.

---

## 📃 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
