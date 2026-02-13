# !pip install lightgbm

# %% [cell]
#1: Imports

# Core
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Utils
from math import radians, cos, sin, asin, sqrt, atan2

# ML
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
import lightgbm as lgb

# Warnings
import warnings
warnings.filterwarnings('ignore')


# %% [cell]
# 2: Load Dataset
df = pd.read_csv("E:\INTTRVU_HACKATHON\dataset.csv")
print("Shape:", df.shape)
display(df.head())
df.info()

# %% [cell]
# Column standardization
df.columns = (
    df.columns.str.strip().str.lower().str.replace(" ", "_")
)

df.rename(columns={"delivery_time_taken(min)": "delivery_time"}, inplace=True)

# Drop id
df.drop(columns=['id'], inplace=True, errors='ignore')

# Extract city
df['city'] = df['delivery_person_id'].str.extract(r'([A-Z]+)')
df.drop(columns=['delivery_person_id'], inplace=True)

# Clean categoricals
df['type_of_order'] = df['type_of_order'].str.strip().str.lower()
df['type_of_vehicle'] = df['type_of_vehicle'].str.strip().str.lower()

# Ratings clamp
df['delivery_person_ratings'] = df['delivery_person_ratings'].clip(1,5)

# Handle geo zeros
geo_cols = [
    'restaurant_latitude','restaurant_longitude',
    'delivery_location_latitude','delivery_location_longitude'
]
df = df[~(df[geo_cols] == 0).any(axis=1)]

# Missing handling
num_cols = ['delivery_person_age','delivery_person_ratings']
cat_cols = ['type_of_order','type_of_vehicle','city']

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
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


# Important
lat_mask = (
    np.sign(df['restaurant_latitude']) != np.sign(df['delivery_location_latitude'])
)
lon_mask = (
    np.sign(df['restaurant_longitude']) != np.sign(df['delivery_location_longitude'])
)

df.loc[lat_mask, 'restaurant_latitude'] = abs(df.loc[lat_mask, 'restaurant_latitude'])
df.loc[lon_mask, 'restaurant_longitude'] = abs(df.loc[lon_mask, 'restaurant_longitude'])


# %% [cell]
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

df['distance_km'] = df.apply(lambda x: haversine(
    x['restaurant_latitude'], x['restaurant_longitude'],
    x['delivery_location_latitude'], x['delivery_location_longitude']
), axis=1)

# FEATURE ENGINNERING

# 1) Partner efficiency (experience proxy)
df['partner_efficiency'] = (
    df['delivery_person_ratings'] * 2 +
    (df['delivery_person_age'] / df['delivery_person_age'].max())
)

# 2) Vehicle speed prior knowledge
vehicle_speed = {
    'motorcycle': 40,
    'scooter': 35,
    'electric_scooter': 30,
    'bicycle': 15
}
df['vehicle_speed'] = df['type_of_vehicle'].map(vehicle_speed)

# 3) Order complexity
order_complexity = {
    'snack': 1,
    'drinks': 1,
    'meal': 2,
    'buffet': 3
}
df['order_complexity'] = df['type_of_order'].map(order_complexity)

# 5) Age group segmentation
df['age_group'] = pd.cut(
    df['delivery_person_age'],
    bins=[18,25,35,50,65],
    labels=['young','adult','mid_age','senior']
)


# %% [cell]
# ---------- CLEANING & SANITY CHECK FOR ENGINEERED FEATURES ----------

# 1) Distance sanity (remove unrealistic logistics values)
df = df[(df['distance_km'] > 0.5) & (df['distance_km'] < 40)]

# 2) Vehicle speed mapping missing handling
df['vehicle_speed'].fillna(df['vehicle_speed'].median(), inplace=True)

# 3) Order complexity missing handling
df['order_complexity'].fillna(df['order_complexity'].median(), inplace=True)

# 4) Partner efficiency normalization (balanced scale)
df['partner_efficiency'] = (
    (df['delivery_person_ratings'] / 5) * 0.7 +
    (df['delivery_person_age'] / 65) * 0.3
)

# 5) Delivery speed proxy cleanup
df['speed_km_per_min'] = df['distance_km'] / df['delivery_time']
df['speed_km_per_min'].replace([np.inf, -np.inf], np.nan, inplace=True)
df['speed_km_per_min'].fillna(df['speed_km_per_min'].median(), inplace=True)

# 6) Age group categorical safety
df['age_group'] = df['age_group'].astype(str)
df['age_group'].replace('nan', 'adult', inplace=True)

# 7) Final duplicate check after feature creation
df.drop_duplicates(inplace=True)

# 8) Final null audit
print(df.isnull().sum())
print("Shape after feature cleaning:", df.shape)


# %% [cell]
# ---------- EDA 2 : FEATURE ENGINEERING IMPACT (MODEL-DRIVEN INSIGHTS) ----------

fig, axes = plt.subplots(4, 2, figsize=(16, 22))
fig.suptitle("EDA 2 — Engineered Feature Insights vs Delivery Time", fontsize=20)

# 1️⃣ Distance vs Delivery Time
sns.scatterplot(x='distance_km', y='delivery_time', data=df, ax=axes[0,0])
axes[0,0].set_title("Distance vs Delivery Time")

# 2️⃣ Partner Efficiency vs Delivery Time
sns.scatterplot(x='partner_efficiency', y='delivery_time', data=df, ax=axes[0,1])
axes[0,1].set_title("Partner Efficiency vs Delivery Time")

# 3️⃣ Speed Proxy vs Delivery Time
sns.scatterplot(x='speed_km_per_min', y='delivery_time', data=df, ax=axes[1,0])
axes[1,0].set_title("Speed vs Delivery Time")

# 4️⃣ Vehicle Speed Impact
sns.boxplot(x='vehicle_speed', y='delivery_time', data=df, ax=axes[1,1])
axes[1,1].set_title("Vehicle Speed vs Delivery Time")

# 5️⃣ Order Complexity Impact
sns.boxplot(x='order_complexity', y='delivery_time', data=df, ax=axes[2,0])
axes[2,0].set_title("Order Complexity vs Delivery Time")

# 6️⃣ Age Group Impact
sns.barplot(x='age_group', y='delivery_time', data=df, ax=axes[2,1])
axes[2,1].set_title("Age Group vs Delivery Time")

# 7️⃣ City Performance
top_cities = df['city'].value_counts().nlargest(10).index
sns.barplot(x='city', y='delivery_time', data=df[df['city'].isin(top_cities)], ax=axes[3,0])
axes[3,0].set_title("Top Cities vs Delivery Time")
axes[3,0].tick_params(axis='x', rotation=45)

# 8️⃣ Feature Correlation (Engineered + Core)
important_cols = [
    'delivery_time','distance_km','partner_efficiency',
    'vehicle_speed','order_complexity','speed_km_per_min',
    'delivery_person_age','delivery_person_ratings'
]
sns.heatmap(df[important_cols].corr(), annot=True, cmap='coolwarm', ax=axes[3,1])
axes[3,1].set_title("Feature Correlation Matrix")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


# %% [cell]
# Target
y = df['delivery_time']

X = df.drop(columns='delivery_time')

categorical_cols = ['type_of_vehicle','type_of_order','city','age_group']
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape, X_test.shape)

# %% [cell]
# Step 4 — Baseline model (must for hackathon)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)


# %% [cell]
# Step 5 — Evaluation metrics
def evaluate(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{model_name} RMSE:", rmse)
    print(f"{model_name} MAE:", mae)
    
evaluate(y_test, lr_pred, "Linear Regression")

# %% [cell]
lr = LinearRegression()
lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)

print("LR RMSE:", np.sqrt(mean_squared_error(y_test, pred_lr)))
print("LR MAE:", mean_absolute_error(y_test, pred_lr))


# %% [cell]

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

evaluate(y_test, rf_pred, "Random Forest")


# %% [cell]
xgb_model = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

evaluate(y_test, xgb_pred, "XGBoost")


# %% [cell]
    
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=10,
        random_state=42
    )
    
    lgb_model.fit(X_train, y_train)
    
    lgb_pred = lgb_model.predict(X_test)
    
    evaluate(y_test, lgb_pred, "LightGBM")


# %% [cell]
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Assuming lr_pred, rf_pred, xgb_pred, lgb_pred are your model predictions
results = pd.DataFrame({
    "Model": ["Linear(Base Model)", "RandomForest", "XGBoost", "LightGBM"],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, lr_pred)),
        np.sqrt(mean_squared_error(y_test, rf_pred)),
        np.sqrt(mean_squared_error(y_test, xgb_pred)),
        np.sqrt(mean_squared_error(y_test, lgb_pred))
    ],
    "MAE": [
        mean_absolute_error(y_test, lr_pred),
        mean_absolute_error(y_test, rf_pred),
        mean_absolute_error(y_test, xgb_pred),
        mean_absolute_error(y_test, lgb_pred)
    ],
    "R²": [
        r2_score(y_test, lr_pred),
        r2_score(y_test, rf_pred),
        r2_score(y_test, xgb_pred),
        r2_score(y_test, lgb_pred)
    ]
})

# Sort by RMSE
results_sorted = results.sort_values(by="RMSE")
print(results_sorted)

# %% [cell]
