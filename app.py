import streamlit as st
import pandas as pd
import numpy as np
import joblib
from math import radians, sin, cos, sqrt, atan2

# ==============================================================================
# CONFIG & SETUP
# ==============================================================================
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        padding: 0.5rem;
    }
    .metric-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# MODEL LOADING
# ==============================================================================
@st.cache_resource
def load_model():
    """Loads the trained model pipeline."""
    try:
        model = joblib.load('best_delivery_time_model.pkl')
        return model
    except FileNotFoundError:
        st.error("Model file 'best_delivery_time_model.pkl' not found. Please ensure it is in the same directory.")
        return None

try:
    model_pipeline = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_pipeline = None

# ==============================================================================
# FEATURE ENGINEERING LOGIC (Replicated from Notebook)
# ==============================================================================
def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculates Haversine distance in km."""
    # Approximate radius of earth in km
    R = 6371.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def preprocess_input(data):
    """
    Apply feature engineering logic exactly as done in the prototype notebook.
    """
    df = pd.DataFrame([data])
    
    # 1. Distance Calculation (Haversine)
    df['distance_km'] = df.apply(lambda x: calculate_distance(
        x['restaurant_latitude'], x['restaurant_longitude'],
        x['delivery_location_latitude'], x['delivery_location_longitude']
    ), axis=1)
    
    # 2. Distance Transformations
    df['distance_log'] = np.log1p(df['distance_km'])
    df['distance_sq'] = df['distance_km'] ** 2
    
    # 3. Manhattan Distance (Simplified as per notebook)
    df['manhattan_km'] = (abs(df['restaurant_latitude'] - df['delivery_location_latitude']) + 
                          abs(df['restaurant_longitude'] - df['delivery_location_longitude']))
    
    # 4. Feature Interactions & Mappings
    
    # Vehicle Score Mapping
    vehicle_map = {
        "motorcycle": 3,
        "scooter": 2,
        "electric_scooter": 2,
        "bicycle": 1
    }
    # Default to 2 if unknown (though UI restricts this)
    df['vehicle_score'] = df['type_of_vehicle'].map(vehicle_map).fillna(2)
    
    # Order Complexity Mapping
    order_map = {
        "Snack": 1,
        "Drinks": 1,
        "Buffet": 2,
        "Meal": 3
    }
    df['order_complexity'] = df['type_of_order'].map(order_map).fillna(2)
    
    # Partner Efficiency
    df['partner_efficiency'] = df['delivery_person_ratings'] * df['multiple_deliveries']
    
    # Distance Interactions
    df['distance_vehicle_interaction'] = df['distance_km'] * df['vehicle_score']
    df['distance_rating_interaction'] = df['distance_km'] * df['delivery_person_ratings']
    
    # 5. Age Bucket (Categorical)
    # bins=[18, 25, 35, 60] -> labels=["young", "mid", "senior"]
    # We use pd.cut logic manually or via pandas to ensure safety with single row
    age = df['delivery_person_age'].iloc[0]
    if 18 <= age <= 25:
        df['age_bucket'] = 'young'
    elif 25 < age <= 35:
        df['age_bucket'] = 'mid'
    else:
        df['age_bucket'] = 'senior' # Covers >35 up to 60+
        
    # Ensure columns derived are correct types (float/int)
    cols_to_float = ['distance_km', 'distance_log', 'distance_sq', 'manhattan_km', 
                     'partner_efficiency', 'vehicle_score', 'order_complexity', 
                     'distance_vehicle_interaction', 'distance_rating_interaction']
    for col in cols_to_float:
        df[col] = df[col].astype(float)
        
    return df

# ==============================================================================
# UI LAYOUT
# ==============================================================================
st.title("🛵 Delivery Time Prediction App")
st.markdown("Enter delivery details to get an estimated delivery time.")

with st.sidebar:
    st.header("📋 Delivery Details")
    
    st.subheader("Delivery Partner")
    age = st.slider("Delivery Person Age", 18, 65, 25)
    ratings = st.slider("Delivery Person Ratings", 1.0, 5.0, 4.5, 0.1)
    # Multiple Deliveries removed from UI as per user request, defaulting to 0 internally
    
    st.subheader("Order Info")
    vehicle_type = st.selectbox("Type of Vehicle", ["motorcycle", "scooter", "electric_scooter", "bicycle"])
    order_type = st.selectbox("Type of Order", ["Snack", "Drinks", "Buffet", "Meal"])
    
    st.subheader("Locations (Coordinates)")
    # Defaults for a sample location
    rest_lat = st.number_input("Restaurant Latitude", value=22.745049, format="%.6f")
    rest_lon = st.number_input("Restaurant Longitude", value=75.892471, format="%.6f")
    del_lat = st.number_input("Delivery Latitude", value=22.765049, format="%.6f")
    del_lon = st.number_input("Delivery Longitude", value=75.912471, format="%.6f")
    
    predict_btn = st.button("Predict Time 🚀")

# ==============================================================================
# MAIN LOGIC
# ==============================================================================

if predict_btn and model_pipeline:
    with st.spinner("Calculating delivery time..."):
        # 1. Prepare raw input data
        raw_data = {
            'delivery_person_age': age,
            'delivery_person_ratings': ratings,
            'multiple_deliveries': 0, # Hardcoded default: 0 (Single order)
            'type_of_vehicle': vehicle_type,
            'type_of_order': order_type,
            'restaurant_latitude': rest_lat,
            'restaurant_longitude': rest_lon,
            'delivery_location_latitude': del_lat,
            'delivery_location_longitude': del_lon
        }
        
        # 2. Preprocess / Feature Engineering
        input_df = preprocess_input(raw_data)
        
        # 3. Predict
        try:
            # We must ensure the column order matches if possible, but sklearn pipelines usually align by name
            # columns expected by pipeline:
            # Categorical: 'type_of_order', 'type_of_vehicle', 'age_bucket'
            # Numerical: includes 'distance_km', 'distance_log', etc.
            
            prediction = model_pipeline.predict(input_df)
            predicted_time = prediction[0]
            
            # 4. Display Results
            st.success("Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-container"><h3>Predicted Time</h3><h1 style="color: #ff4b4b;">{:.2f} min</h1></div>'.format(predicted_time), unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="metric-container"><h3>Distance</h3><h2>{:.2f} km</h2></div>'.format(input_df['distance_km'][0]), unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="metric-container"><h3>Vehicle Score</h3><h2>{:.0f}</h2></div>'.format(input_df['vehicle_score'][0]), unsafe_allow_html=True)
            
            # Detailed Factors Expander
            with st.expander("See factors affecting this prediction"):
                st.write("These calculated features were passed to the model:")
                
                # FIX: Only apply float formatting to numeric columns
                numeric_cols = input_df.select_dtypes(include=[np.number]).columns
                st.dataframe(input_df[numeric_cols].T.style.format("{:.2f}"))
                
                # Display categorical fields separately without formatting
                cat_cols = input_df.select_dtypes(exclude=[np.number]).columns
                if len(cat_cols) > 0:
                    st.write("Categorical Inputs:")
                    st.dataframe(input_df[cat_cols].T)

            # ==============================================================================
            # VISUALIZATIONS
            # ==============================================================================
            st.markdown("---")
            st.header("📊 Comparative Analysis")
            
            @st.cache_data
            def load_data():
                try:
                    df = pd.read_csv('cleaned_dataset_fooddelivery.csv')
                    return df
                except FileNotFoundError:
                    return None

            hist_data = load_data()
            
            if hist_data is not None:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                col_graph1, col_graph2 = st.columns(2)
                
                with col_graph1:
                    st.subheader("Time Distribution")
                    fig1, ax1 = plt.subplots(figsize=(6, 4))
                    sns.histplot(hist_data['delivery_time'], kde=True, ax=ax1, color='#FF4B4B', alpha=0.5)
                    ax1.axvline(predicted_time, color='blue', linestyle='--', linewidth=2, label='Your Order')
                    ax1.set_title(f"Prediction vs Historical Data")
                    ax1.set_xlabel("Delivery Time (min)")
                    ax1.legend()
                    st.pyplot(fig1)
                    
                with col_graph2:
                    st.subheader("Impact of Vehicle")
                    # Filter for reasonable range if outliers exist
                    fig2, ax2 = plt.subplots(figsize=(6, 4))
                    sns.boxplot(x='type_of_vehicle', y='delivery_time', data=hist_data, ax=ax2, palette="Set2")
                    # Highlight current prediction range roughly (just a horizontal line)
                    ax2.axhline(predicted_time, color='blue', linestyle='--', linewidth=2, label='Predicted')
                    ax2.set_title("Delivery Time by Vehicle")
                    ax2.legend()
                    st.pyplot(fig2)
                    
                # Extra: Feature Importance (Static/Conceptual if model doesn't support easy access in pipeline)
                # Since we used a pipeline, accessing importance is a bit tricky but possible if step is named 'model'
                # Let's try to get feature importance if possible
                if hasattr(model_pipeline.named_steps['model'], 'feature_importances_'):
                    st.subheader("Feature Importance")
                    feature_names = numeric_cols.tolist() # Simplified assumption since OneHotEncoder adds names
                    # We might skip exact names due to complexity of OneHot output matching
                    # Instead, we iterate on the conceptual chart from before but better
                    
                    st.info("Visualizing where your delivery stands compared to averages:")
                    
                    avg_time_vehicle = hist_data[hist_data['type_of_vehicle'] == vehicle_type]['delivery_time'].mean()
                    
                    metrics_comp = pd.DataFrame({
                        "Metric": ["Your Prediction", f"Avg for {vehicle_type}"],
                        "Time (min)": [predicted_time, avg_time_vehicle]
                    })
                    st.bar_chart(metrics_comp.set_index("Metric"))
                    
            else:
                st.warning("Historical training data (cleaned_dataset_fooddelivery.csv) not found. Graphs cannot be generated.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            # st.write("Debug - Input DataFrame columns:", input_df.columns.tolist())

elif not model_pipeline:
    st.warning("Model pipeline is not loaded. Cannot predict.")
else:
    st.info("Adjust parameters in the sidebar and click 'Predict Time' to see results.")
