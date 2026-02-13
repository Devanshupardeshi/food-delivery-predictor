# 🛵 Food Delivery Time Predictor

A machine learning-powered web application that predicts food delivery times based on delivery partner details, order characteristics, and location data. Built with **Streamlit** and **LightGBM**.

---

## 🌟 App Features

### 🎯 Accurate Predictions

- **Real-time Inference**: Instantly calculates estimated delivery time upon user input.
- **Interactive Sidebar**: Adjust Delivery Partner Age, Ratings, Vehicle Type, and Geo-coordinates easily.

### 📊 Advanced Data Visualization

The app includes a **Comparative Analysis** section to contextualize the prediction:

- **Time Distribution Analysis**: A histogram comparing your predicted time against the historical distribution of all delivery times.
- **Vehicle Impact Chart**: A boxplot showing how different vehicle types (Motorcycle, Scooter, Bicycle) affect delivery speed.
- **Feature Breakdown**: Detailed dataframe view of the engineered features sent to the model (Distance, Interaction Scores, etc.).

---

## 🔬 Methodology for Best Solution

To achieve the highest prediction accuracy (Root Mean Squared Error of ~7.23), we implemented a rigorous data processing and feature engineering pipeline derived from extensive analysis in `prototype.ipynb`.

### 1. Data Preprocessing

Before feeding data into the model, raw inputs undergo several cleaning steps:

- **Missing Value Imputation**: Handling potential missing data points with median/mode strategies (implicit in the pipeline logic).
- **Categorical Handling**:
  - **One-Hot Encoding**: Used for `type_of_order` and `type_of_vehicle` to allow the model to understand categorical distinctions without imposing arbitrary ordinality.
  - **Binning**: `delivery_person_age` is discretized into buckets ('young', 'mid', 'senior') to capture non-linear age-related trends.

### 2. Advanced Feature Engineering

We didn't just use raw columns; we created new, high-signal features:

- **Geospatial Distance Calculation**:
  - **Haversine Distance**: Calculates the precise "as-the-crow-flies" distance between restaurant and delivery coordinates on the Earth's surface.
  - **Manhattan Distance**: A secondary metric (`abs(lat_diff) + abs(lon_diff)`) that often better approximates city grid travel.
- **Distance Transformations**:
  - `distance_log`: Applying `np.log1p` to distances to normalize skewed distributions and reduce the impact of outliers.
  - `distance_sq`: Squaring distance to capture exponential relationships (e.g., traffic delays compounding on longer routes).
- **Interaction Features**:
  - **Partner Efficiency**: Combines `ratings` and `multiple_deliveries` to model how driver experience and workload interact.
  - **Distance-Vehicle Interaction**: Multiplies distance by a mapped `vehicle_score` (Motorcycle=3, Bicycle=1) to account for speed differences over varied distances.
  - **Distance-Rating Interaction**: Models how highly-rated drivers might manage longer routes more efficiently.

### 3. Model Architecture

- **Algorithm**: **LightGBM Regressor** (Gradient Boosting Machine).
- **Why LightGBM?**: Chosen for its superior speed and efficiency with large datasets, and its ability to handle complex non-linear feature interactions better than traditional linear models.
- **Pipeline Integration**: The entire preprocessing and prediction flow is encapsulated in a Scikit-Learn `Pipeline`, ensuring that the exact same transformations applied during training are applied during real-time inference in the app.

### 4. Model Evaluation Metrics

The selected LightGBM model outperformed other candidates (Linear Regression, Random Forest, XGBoost) during cross-validation:

| Metric         | Score           | Description                                                                                                                       |
| :------------- | :-------------- | :-------------------------------------------------------------------------------------------------------------------------------- |
| **RMSE** | **~7.23** | Root Mean Squared Error (Lower is better). Indicates the model's predictions are typically within ±7 minutes of the actual time. |
| **R²**  | **~0.82** | R-Squared (Higher is better). Explains approx. 82% of the variance in delivery times.                                             |

---

## 🚀 Setup & Installation

### Prerequisites

- Python 3.8+
- pip

### 1. Clone & Install

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Logic (Optional)

Run the headless verification script to test the model pipeline without launching the UI:

```bash
python verify_app.py
```

### 3. Run Locally

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

### 3. Deploy to Render

1. Create a new**Web Service** on [Render](https://render.com/).
2. Connect this repository.
3. Use the following settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py`

---

## 📂 Project Structure

- `app.py`: Main application code containing UI, Feature Engineering logic, and Visualization code.
- `requirements.txt`: Python package dependencies.
- `best_delivery_time_model.pkl`: Pre-trained model pipeline (required).
- `cleaned_dataset_fooddelivery.csv`: Dataset used for generating comparative visualizations.

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Engine**: Scikit-Learn, LightGBM
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
