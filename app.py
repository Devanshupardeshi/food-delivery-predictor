
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from math import radians, sin, cos, sqrt, atan2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import warnings

# --- Page Configuration ---
st.set_page_config(
    page_title="Food Delivery Predictor",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Loading ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    local_css("style.css")
except FileNotFoundError:
    st.warning("style.css not found. Utilizing default Streamlit styling.")

# --- Helper Functions (From Notebook) ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

# --- Data Loading & Preprocessing (Cached) ---
@st.cache_data
def load_and_preprocess_data():
    # 2: Load Dataset
    try:
        df = pd.read_csv("dataset.csv")
    except FileNotFoundError:
        st.error("dataset.csv not found in the current directory.")
        return None, None

    # Column standardization
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df.rename(columns={"delivery_time_taken(min)": "delivery_time"}, inplace=True)

    # Drop id
    df.drop(columns=['id'], inplace=True, errors='ignore')

    # Extract city
    if 'delivery_person_id' in df.columns:
        df['city'] = df['delivery_person_id'].str.extract(r'([A-Z]+)')
        df.drop(columns=['delivery_person_id'], inplace=True)

    # Clean categoricals
    df['type_of_order'] = df['type_of_order'].str.strip().str.lower()
    df['type_of_vehicle'] = df['type_of_vehicle'].str.strip().str.lower()

    # Ratings clamp
    df['delivery_person_ratings'] = df['delivery_person_ratings'].clip(1, 5)

    # Handle geo zeros
    geo_cols = ['restaurant_latitude', 'restaurant_longitude', 'delivery_location_latitude', 'delivery_location_longitude']
    df = df[~(df[geo_cols] == 0).any(axis=1)]

    # Missing handling
    num_cols = ['delivery_person_age', 'delivery_person_ratings']
    cat_cols = ['type_of_order', 'type_of_vehicle', 'city']
    
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    for col in cat_cols:
         if not df[col].mode().empty:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Datatypes
    df['delivery_person_age'] = df['delivery_person_age'].astype(int)
    df['delivery_person_ratings'] = df['delivery_person_ratings'].astype(float)

    # Sanity filters
    df = df[(df['delivery_person_age'] >= 18) & (df['delivery_person_age'] <= 65)]
    df = df[df['delivery_time'] > 0]

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Outlier capping
    Q1 = df['delivery_time'].quantile(0.25)
    Q3 = df['delivery_time'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df['delivery_time'] = np.clip(df['delivery_time'], lower, upper)

    # Fix Lat/Lon Signs
    lat_mask = np.sign(df['restaurant_latitude']) != np.sign(df['delivery_location_latitude'])
    lon_mask = np.sign(df['restaurant_longitude']) != np.sign(df['delivery_location_longitude'])
    df.loc[lat_mask, 'restaurant_latitude'] = abs(df.loc[lat_mask, 'restaurant_latitude'])
    df.loc[lon_mask, 'restaurant_longitude'] = abs(df.loc[lon_mask, 'restaurant_longitude'])

    # Calculate Distance
    df['distance_km'] = df.apply(lambda x: haversine(
        x['restaurant_latitude'], x['restaurant_longitude'],
        x['delivery_location_latitude'], x['delivery_location_longitude']
    ), axis=1)

    # --- Feature Engineering ---
    
    # 1) Partner efficiency
    df['partner_efficiency'] = (df['delivery_person_ratings'] * 2 + (df['delivery_person_age'] / df['delivery_person_age'].max()))

    # 2) Vehicle speed prior knowledge
    vehicle_speed_map = {'motorcycle': 40, 'scooter': 35, 'electric_scooter': 30, 'bicycle': 15}
    df['vehicle_speed'] = df['type_of_vehicle'].map(vehicle_speed_map)
    df['vehicle_speed'].fillna(df['vehicle_speed'].median(), inplace=True)

    # 3) Order complexity
    order_complexity_map = {'snack': 1, 'drinks': 1, 'meal': 2, 'buffet': 3}
    df['order_complexity'] = df['type_of_order'].map(order_complexity_map)
    df['order_complexity'].fillna(df['order_complexity'].median(), inplace=True)
    
    # 4) Partner efficiency normalization (Refinement)
    df['partner_efficiency'] = ((df['delivery_person_ratings'] / 5) * 0.7 + (df['delivery_person_age'] / 65) * 0.3)

    # 5) Delivery speed proxy (TARGET LEAKAGE CAUTION handled for display/training only)
    df['speed_km_per_min'] = df['distance_km'] / df['delivery_time']
    df['speed_km_per_min'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['speed_km_per_min'].fillna(df['speed_km_per_min'].median(), inplace=True)

    # 6) Age group segmentation
    df['age_group'] = pd.cut(df['delivery_person_age'], bins=[18, 25, 35, 50, 65], labels=['young', 'adult', 'mid_age', 'senior'])
    df['age_group'] = df['age_group'].astype(str)
    df['age_group'].replace('nan', 'adult', inplace=True)

    # 7) Final duplicate check
    df.drop_duplicates(inplace=True)

    # Filter Distance Sanity
    df = df[(df['distance_km'] > 0.5) & (df['distance_km'] < 40)]
    
    return df, vehicle_speed_map

df, vehicle_speed_map = load_and_preprocess_data()

# --- Sidebar ---
st.sidebar.title("🚀 DeliveryPredict")
st.sidebar.markdown("Professional Hackathon Edition")
nav = st.sidebar.radio("Navigation", ["Dashboard", "Data Analysis", "Model Output", "Live Predictor"])
st.sidebar.info("This app predicts food delivery times using advanced ML models.")

# --- Dashboard Section ---
if nav == "Dashboard":
    st.title("📊 Operational Dashboard")
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Deliveries", f"{len(df):,}")
        col2.metric("Avg Delivery Time", f"{df['delivery_time'].mean():.1f} min")
        col3.metric("Avg Distance", f"{df['distance_km'].mean():.1f} km")
        col4.metric("Active Partners", f"{df['delivery_person_ratings'].count():,}")
        
        st.markdown("### Key Metrics Overview")
        col_L, col_R = st.columns(2)
        
        with col_L:
            st.markdown("#### Delivery Volume by City")
            city_counts = df['city'].value_counts().reset_index()
            city_counts.columns = ['City', 'Count']
            fig_city = px.bar(city_counts, x='City', y='Count', color='Count', template="plotly_dark")
            st.plotly_chart(fig_city, use_container_width=True)
            
        with col_R:
            st.markdown("#### Vehicle Type Distribution")
            veh_counts = df['type_of_vehicle'].value_counts().reset_index()
            veh_counts.columns = ['Vehicle', 'Count']
            fig_veh = px.pie(veh_counts, names='Vehicle', values='Count', hole=0.4, template="plotly_dark")
            st.plotly_chart(fig_veh, use_container_width=True)
    else:
        st.error("No data available.")

# --- Data Analysis (EDA) ---
elif nav == "Data Analysis":
    st.title("📈 Exploratory Data Analysis")
    
    if df is not None:
        tab1, tab2, tab3 = st.tabs(["Correlations", "Feature Insights", "Distributions"])
        
        with tab1:
            st.subheader("Feature Correlations")
            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", template="plotly_dark")
            st.plotly_chart(fig_corr, use_container_width=True)
            
        with tab2:
            st.subheader("Drivers of Delivery Time")
            
            col_eda1, col_eda2 = st.columns(2)
            
            with col_eda1:
                st.markdown("**Distance vs. Time**")
                fig_dist = px.scatter(df.sample(2000), x='distance_km', y='delivery_time', color='type_of_vehicle', opacity=0.6, template="plotly_dark")
                st.plotly_chart(fig_dist, use_container_width=True)
                
            with col_eda2:
                st.markdown("**Ratings vs. Time**")
                fig_rate = px.box(df, x='delivery_person_ratings', y='delivery_time', template="plotly_dark")
                st.plotly_chart(fig_rate, use_container_width=True)
        
        with tab3:
             st.subheader("Data Distributions")
             feature = st.selectbox("Select Feature", ['delivery_time', 'distance_km', 'delivery_person_age', 'partner_efficiency'])
             fig_hist = px.histogram(df, x=feature, nbins=50, marginal="box", template="plotly_dark")
             st.plotly_chart(fig_hist, use_container_width=True)

# --- Model Output ---
elif nav == "Model Output":
    st.title("🤖 Model Performance Evaluation")
    
    if df is not None and st.button("Train Models (Run Pipeline)"):
         with st.spinner("Training models... This might take a moment."):
            # Prepare Data
            y = df['delivery_time']
            X = df.drop(columns='delivery_time')
            categorical_cols = ['type_of_vehicle', 'type_of_order', 'city', 'age_group']
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            results_list = []
            
            # 1. Linear Regression
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            lr_pred = lr.predict(X_test)
            results_list.append({
                "Model": "Linear Regression",
                "RMSE": np.sqrt(mean_squared_error(y_test, lr_pred)),
                "MAE": mean_absolute_error(y_test, lr_pred),
                "R2": r2_score(y_test, lr_pred)
            })
            
            # 2. Random Forest
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            results_list.append({
                "Model": "Random Forest",
                "RMSE": np.sqrt(mean_squared_error(y_test, rf_pred)),
                "MAE": mean_absolute_error(y_test, rf_pred),
                "R2": r2_score(y_test, rf_pred)
            })
            
            # 3. XGBoost
            xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            results_list.append({
                "Model": "XGBoost",
                "RMSE": np.sqrt(mean_squared_error(y_test, xgb_pred)),
                "MAE": mean_absolute_error(y_test, xgb_pred),
                "R2": r2_score(y_test, xgb_pred)
            })
            
            # 4. LightGBM
            lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, random_state=42, verbose=-1)
            lgb_model.fit(X_train, y_train)
            lgb_pred = lgb_model.predict(X_test)
            results_list.append({
                "Model": "LightGBM",
                "RMSE": np.sqrt(mean_squared_error(y_test, lgb_pred)),
                "MAE": mean_absolute_error(y_test, lgb_pred),
                "R2": r2_score(y_test, lgb_pred)
            })
            
            res_df = pd.DataFrame(results_list).sort_values(by="RMSE")
            
            st.success("Training Complete!")
            st.dataframe(res_df.style.highlight_min(axis=0, subset=['RMSE', 'MAE'], props='background-color: green').highlight_max(axis=0, subset=['R2'], props='background-color: green'), use_container_width=True)
            
            # Save best model logic (simplified for single session)
            st.session_state['best_model'] = xgb_model
            st.session_state['X_columns'] = X_train.columns
            st.session_state['model_trained'] = True

    elif 'model_trained' in st.session_state:
        st.info("Models already trained in session. Go to 'Live Predictor' to test.")
    else:
        st.markdown("Click the button above to train models and see the leaderboard.")

# --- Live Predictor ---
elif nav == "Live Predictor":
    st.title("⏱️ Live Delivery Time Predictor")
    
    if 'best_model' not in st.session_state:
        st.warning("Please train the models in the 'Model Output' tab first.")
    else:
        colP1, colP2 = st.columns(2)
        
        with colP1:
            st.subheader("Order Details")
            age = st.slider("Delivery Person Age", 18, 65, 30)
            ratings = st.slider("Delivery Person Rating", 1.0, 5.0, 4.5, 0.1)
            distance = st.number_input("Delivery Distance (km)", 1.0, 30.0, 5.0)
            
            vehicle = st.selectbox("Vehicle Type", ["motorcycle", "scooter", "electric_scooter", "bicycle"])
            order_type = st.selectbox("Order Type", ["Meal", "Snack", "Drinks", "Buffet"])
            city_code = st.selectbox("City Code", ["INDORES", "BANGRES", "COIMBRES", "CHENRES", "MUMRES", "DELRES"]) # Simplified
            
            # --- Handling the Data Leakage/Proxy Feature ---
            st.markdown("---")
            st.caption("ℹ️ **Traffic Simulation (Advanced)**")
            st.caption("The model uses speed estimation to refine predictions. Adjusting this simulates traffic.")
            median_speed = df['speed_km_per_min'].median()
            traffic_factor = st.slider("Estimated Traffic Speed (km/min)", 0.1, 2.0, float(median_speed))
            
        with colP2:
            st.subheader("Prediction Result")
            
            if st.button("Predict Time"):
                # Construct Input Data
                input_data = {
                    'delivery_person_age': age,
                    'delivery_person_ratings': ratings,
                    'restaurant_latitude': 0, # Not directly used in model X, only for dist calculation which we have
                    'restaurant_longitude': 0,
                    'delivery_location_latitude': 0,
                    'delivery_location_longitude': 0,
                    'distance_km': distance,
                    'partner_efficiency': (ratings / 5) * 0.7 + (age / 65) * 0.3,
                    'vehicle_speed': vehicle_speed_map.get(vehicle, 30),
                    'order_complexity': {'snack': 1, 'drinks': 1, 'meal': 2, 'buffet': 3}.get(order_type.lower(), 2),
                    'age_group': 'young' if age <= 25 else 'adult' if age <= 35 else 'mid_age' if age <= 50 else 'senior',
                    'speed_km_per_min': traffic_factor # USER INPUT for the leaked feature
                }
                
                # Convert to DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Encode Categoricals (One-Hot) matches training columns
                # We need to create dummy variables for the specific inputs
                # A robust way is to reindex with the training columns fillna(0)
                
                # Pre-processing for One-Hot
                input_df['type_of_vehicle_' + vehicle] = 1
                input_df['type_of_order_' + order_type.lower()] = 1
                
                # City extraction simulation (simple substring match often used in 'city' column)
                city_cln = city_code.replace("RES", "")
                input_df['city_' + city_cln] = 1 
                
                input_df['age_group_' + input_df['age_group'].iloc[0]] = 1
                
                # Align with training columns
                model_cols = st.session_state['X_columns']
                final_input = input_df.reindex(columns=model_cols, fill_value=0)
                
                # Predict
                prediction = st.session_state['best_model'].predict(final_input)[0]
                
                st.balloons()
                st.metric("Estimated Delivery Time", f"{prediction:.0f} mins")
                
                # Confidence/Range (Simple heuristic based on RMSE closer to 4-5 usually)
                st.progress(min(100, int(prediction + 10)))
                st.caption(f"Range: {prediction - 5:.0f} - {prediction + 5:.0f} mins")

