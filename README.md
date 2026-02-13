# 🍔 Food Delivery Time Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A professional, data-driven web application to predict food delivery times based on various operational factors. Built for the Hackathon 2026.

## 🚀 Features

*   **Operational Dashboard**: Real-time overview of key metrics (Total deliveries, active partners, city-wise breakdown).
*   **Interactive EDA**: Deep dive into data with interactive charts showing correlations, driver performance, and vehicle impact.
*   **Advanced Logic**: Implements custom feature engineering (Haversine distance, partner efficiency scores, order complexity).
*   **Multi-Model Training**: Train and compare **Linear Regression, Random Forest, XGBoost, and LightGBM** directly from the UI.
*   **Live Prediction**: "What-If" analysis tool to predict delivery times for new orders, featuring traffic simulation.

## 🛠️ Tech Stack

*   **Frontend**: [Streamlit](https://streamlit.io/) (with custom CSS for Glassmorphism UI)
*   **Data Processing**: Pandas, NumPy
*   **Visualization**: Plotly, Seaborn, Matplotlib
*   **Machine Learning**: Scikit-Learn, XGBoost, LightGBM

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/food-delivery-predictor.git
    cd food-delivery-predictor
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app**:
    ```bash
    streamlit run app.py
    ```

## 📊 Model Performance

| Model | RMSE | MAE | R² Score |
| :--- | :--- | :--- | :--- |
| **LightGBM** | ~4.5 | ~3.8 | 0.82 |
| **XGBoost** | ~4.7 | ~3.9 | 0.80 |
| **Random Forest** | ~5.1 | ~4.1 | 0.78 |
| **Linear Regression** | ~6.2 | ~5.0 | 0.65 |

*(Metrics are indicative based on the prototype dataset)*

## 📂 Project Structure

```
├── app.py                # Main Streamlit application
├── style.css             # Custom styling (Glassmorphism theme)
├── dataset.csv           # Training dataset
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## 🤝 Contribution

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

[MIT](https://choosealicense.com/licenses/mit/)
