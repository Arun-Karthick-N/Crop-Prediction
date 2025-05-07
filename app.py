import pandas as pd
import numpy as np
import streamlit as st
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Constants
CROP_TYPES = ['RICE', 'WHEAT', 'SORGHUM', 'PEARL MILLET', 'MAIZE',
              'CHICKPEA', 'PIGEONPEA', 'GROUNDNUT', 'SESAMUM',
              'RAPESEED AND MUSTARD', 'SOYABEAN', 'COTTON']

# Load dataset
@st.cache_data
def load_data():
    file_path = "ICRISAT-District Level Data.csv"
    if not os.path.exists(file_path):
        st.error(f"Dataset not found at {file_path}")
        st.stop()
    df = pd.read_csv(file_path)
    return preprocess_data(df)

# Preprocessing
def preprocess_data(data):
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        data[column] = data[column].fillna(data[column].median())

    data['Year_Group'] = pd.cut(data['Year'], bins=5, labels=False)

    for crop in CROP_TYPES:
        yield_col = f'{crop} YIELD (Kg per ha)'
        area_col = f'{crop} AREA (1000 ha)'
        if yield_col in data.columns and area_col in data.columns:
            data[f'{crop}_PROFIT_INDICATOR'] = data[yield_col] / (data[area_col] + 1)
    return data

# Calculate crop saturation
def calculate_crop_saturation(data, state, district, year, crop_name):
    region_data = data[(data['State Name'] == state) &
                       (data['Dist Name'] == district) &
                       (data['Year'] <= year)]
    if len(region_data) == 0:
        return 0

    latest_year = region_data['Year'].max()
    region_data = region_data[region_data['Year'] == latest_year]
    area_col = f'{crop_name} AREA (1000 ha)'

    if area_col not in region_data.columns or region_data[area_col].isnull().all():
        return 0

    total_crop_columns = [col for col in region_data.columns if 'AREA (1000 ha)' in col]
    total_area = region_data[total_crop_columns].sum(axis=1).iloc[0]
    if total_area == 0:
        return 0

    crop_area = region_data[area_col].iloc[0]
    return (crop_area / total_area) * 100

# Train and predict yield
def predict_yield(data, crop_name, year, state_code, dist_code, area_size):
    yield_col = f'{crop_name} YIELD (Kg per ha)'
    area_col = f'{crop_name} AREA (1000 ha)'

    if yield_col not in data.columns or area_col not in data.columns:
        return None

    model_data = data[[yield_col, area_col, 'Year', 'State Code', 'Dist Code']].dropna()
    if len(model_data) < 100:
        return None

    X = model_data[['Year', 'State Code', 'Dist Code', area_col]]
    y = model_data[yield_col]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    prediction_input = pd.DataFrame({
        'Year': [year],
        'State Code': [state_code],
        'Dist Code': [dist_code],
        area_col: [area_size / 1000]
    })

    return model.predict(prediction_input)[0]

# Crop recommendation logic
def recommend_crop(data, state, district, year, area_size, top_n=10):
    results = []
    region_data = data[(data['State Name'] == state) & (data['Dist Name'] == district)]

    if len(region_data) == 0:
        return pd.DataFrame()

    state_code = region_data['State Code'].iloc[0]
    dist_code = region_data['Dist Code'].iloc[0]

    for crop in CROP_TYPES:
        yield_col = f'{crop} YIELD (Kg per ha)'
        area_col = f'{crop} AREA (1000 ha)'

        if yield_col not in data.columns or area_col not in data.columns:
            continue

        saturation = calculate_crop_saturation(data, state, district, year, crop)
        predicted_yield = predict_yield(data, crop, year, state_code, dist_code, area_size)
        if predicted_yield is None:
            continue

        historical_data = region_data[region_data['Year'] < year]
        profit_indicator = historical_data[f'{crop}_PROFIT_INDICATOR'].mean() if len(historical_data) > 0 and f'{crop}_PROFIT_INDICATOR' in historical_data.columns else 0

        score = saturation * 0.5 - predicted_yield * 0.3 - profit_indicator * 0.2

        results.append({
            'Crop': crop,
            'Saturation': round(saturation, 6),
            'Predicted Yield (kg/ha)': round(predicted_yield, 4),
            'Profit Indicator': round(profit_indicator, 6),
            'Score': round(score, 6)
        })

    return pd.DataFrame(results).sort_values('Score').head(top_n)

# Streamlit App UI
def main():
    st.set_page_config(layout="wide")
    st.title("🌾 Crop Recommendation System")
    st.markdown("This app recommends the best crops for a selected region and year based on yield, saturation, and profitability.")

    data = load_data()

    states = sorted(data['State Name'].unique())
    selected_state = st.selectbox("Select State", states)

    districts = sorted(data[data['State Name'] == selected_state]['Dist Name'].unique())
    selected_district = st.selectbox("Select District", districts)

    year = st.number_input("Enter Year for Prediction", min_value=2000, max_value=2030, value=2020)
    area = st.number_input("Enter Land Area (in acres)", min_value=1.0, max_value=1000.0, value=50.0)

    if st.button("Get Crop Recommendations"):
        with st.spinner("Processing..."):
            recommendations = recommend_crop(data, selected_state, selected_district, year, area)
        
        if recommendations.empty:
            st.warning("No recommendations could be generated for the selected inputs.")
        else:
            st.subheader("Top 10 Crop Recommendations (sorted by score, lower is better):")
            st.dataframe(recommendations, use_container_width=True)

            st.subheader("📊 Predicted Yield Comparison")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=recommendations, x="Crop", y="Predicted Yield (kg/ha)", palette="viridis", ax=ax)
            ax.set_title(f"Predicted Yields in {selected_district}, {selected_state}")
            ax.set_ylabel("Yield (kg/ha)")
            ax.set_xlabel("Crop")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
