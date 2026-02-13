# 🍔 Food Delivery Time Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://devanshupardeshi-food-delivery-predictor-app-1ztjqo.streamlit.app/)
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

## 📊 Model Performance (Actual Run)

| Model | RMSE | MAE | R² Score |
| :--- | :--- | :--- | :--- |
| **XGBoost** | **0.55** | 0.34 | **99.6%** |
| **LightGBM** | 0.59 | 0.39 | 99.5% |
| **Random Forest** | 0.64 | **0.33** | 99.5% |
| **Linear Regression** | 4.59 | 3.41 | 75.8% |

*(Metrics captured from local training run)*

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
