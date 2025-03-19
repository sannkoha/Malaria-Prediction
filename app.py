import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

########################################
# Data Fetching & Preprocessing Functions
########################################

@st.cache_data
def fetch_api_data():
    """
    Fetches data from the API and returns a DataFrame.
    """
    url = "http://hubapi.duchimatech.com/External/pullMLData"
    payload = {
        "_token": "F1A7D842-36B5-4C8C-A2B5-6F9D19D2B073",
        "StartDate": "2023-01-01",  # expanded start date
        "EndDate": "2025-02-28"
    }
    response = requests.post(url, json=payload)
    api_response = response.json()
    if api_response.get('status') != 200:
        st.error("API Error: Status code is not 200.")
        return None
    df_api = pd.DataFrame(api_response.get('data'))
    return df_api

def preprocess_api_data(df_api):
    """
    Converts the DHIS2Period to a Date column, assigns default District names for missing values,
    and extracts MeasurementType and AgeGroup from DataElementName.
    """
    def parse_dhis2period(period_str):
        try:
            year = int(period_str[:4])
            month = int(period_str[4:])
            return pd.Timestamp(year=year, month=month, day=1)
        except Exception as e:
            return pd.NaT

    df_api['Date'] = df_api['DHIS2Period'].apply(parse_dhis2period)
    
    # Ensure missing District values are empty strings and strip extra spaces
    df_api['District'] = df_api['District'].fillna("").apply(lambda x: x.strip())
    # Extract all non-empty district names from the API data
    extracted_districts = df_api['District'][df_api['District'] != ""].unique()
    default_districts = sorted(extracted_districts)
    # If no district is extracted, fallback to a predefined list
    if len(default_districts) == 0:
         default_districts = ['Bo', 'Kenema', 'Port Loko', 'Bombali', 'Kailahun']
    # Assign a default district in a round-robin fashion for empty District entries
    def assign_default_district(val, idx):
        if val == "":
            return default_districts[idx % len(default_districts)]
        else:
            return val
    df_api['District'] = [assign_default_district(val, i) for i, val in enumerate(df_api['District'])]
    
    def parse_data_element(de_name):
        de_clean = de_name.replace("KD_Kush", "").strip()
        m = re.search(r'(.+?)\s*([<>]\d+)', de_clean)
        if m:
            measurement_type = m.group(1).strip()
            age_indicator = m.group(2)
            if age_indicator == '<5':
                age_group = 'lt5'
            elif age_indicator == '>5':
                age_group = 'gt5'
            else:
                age_group = 'unknown'
        else:
            measurement_type = de_clean
            age_group = 'unknown'
        return pd.Series([measurement_type, age_group])
    
    df_api[['MeasurementType', 'AgeGroup']] = df_api['DataElementName'].apply(parse_data_element)
    return df_api

def pivot_and_feature_engineer(df_api):
    """
    Pivots the raw data so each row corresponds to a District and Date,
    creates a single lag feature, and adds month, year, and district dummy features.
    """
    # Pivot data: each row is a District-Date pair; columns for each MeasurementType & AgeGroup
    df_pivot = df_api.pivot_table(
        index=['District', 'Date'],
        columns=['MeasurementType', 'AgeGroup'],
        values='DHIS2AggregateValue',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    
    # Flatten MultiIndex columns
    df_pivot.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_pivot.columns.values]
    
    # Rename District and Date columns for consistency
    if 'District' in df_pivot.columns:
        df_pivot.rename(columns={'District': 'District_'}, inplace=True)
    if 'Date' in df_pivot.columns:
        df_pivot.rename(columns={'Date': 'Date_'}, inplace=True)
    
    # Define target variable (if Victims columns exist, sum them)
    if 'Victims_gt5' in df_pivot.columns and 'Victims_lt5' in df_pivot.columns:
        df_pivot['Total_Victims'] = df_pivot['Victims_lt5'] + df_pivot['Victims_gt5']
        target = 'Total_Victims'
    elif any(col.startswith('Victims') for col in df_pivot.columns):
        target = [col for col in df_pivot.columns if col.startswith('Victims')][0]
    else:
        target = 'DHIS2AggregateValue'
    
    # Sort by District and Date
    df_pivot.sort_values(by=['District_', 'Date_'], inplace=True)
    
    # Create a single lag feature for temporal dependency
    df_pivot['Lag_1'] = df_pivot.groupby('District_')[target].shift(1)
    
    # Add month and year features
    df_pivot['Month'] = df_pivot['Date_'].dt.month
    df_pivot['Year'] = df_pivot['Date_'].dt.year
    
    # One-hot encode the district names (dummy columns with prefix "District_")
    district_dummies = pd.get_dummies(df_pivot['District_'], prefix='District_')
    df_model = pd.concat([df_pivot, district_dummies], axis=1)
    
    # Drop rows with missing lag values (first record per district)
    df_model = df_model.dropna(subset=['Lag_1']).reset_index(drop=True)
    
    # Define the feature columns in the correct order
    feature_cols = ['Lag_1', 'Month', 'Year'] + list(district_dummies.columns)
    
    return df_model, feature_cols, target, list(district_dummies.columns)

@st.cache_resource
def load_model():
    """
    Loads the pre-trained Random Forest model.
    """
    model = joblib.load('best_rf_model_api_ver.pkl')
    return model

@st.cache_data
def get_processed_data():
    """
    Fetches, preprocesses, and feature engineers the API data.
    Returns the modeling DataFrame, feature columns list, target variable, and district dummy columns.
    """
    df_api = fetch_api_data()
    if df_api is None:
        return None, None, None, None
    df_api = preprocess_api_data(df_api)
    df_model, feature_cols, target, dummy_cols = pivot_and_feature_engineer(df_api)
    return df_model, feature_cols, target, dummy_cols

########################################
# Forecasting Function
########################################

def forecast_future(model, df_model, feature_cols, target, dummy_cols, selected_districts, forecast_months, current_override=None):
    """
    For each selected district, this function generates future predictions by iteratively updating the single lag feature.
    If more than one district is selected, the current_override is only applied when a single district is selected.
    """
    all_future_predictions = []
    for district in selected_districts:
        # Filter data for the current district and sort by Date
        df_district = df_model[df_model['District_'] == district].copy()
        if df_district.empty:
            continue
        df_district = df_district.sort_values('Date_')
        last_date = df_district['Date_'].max()
        # Use the override value if provided and only one district is selected; otherwise, use the last target value
        if current_override is not None and len(selected_districts) == 1:
            last_target = current_override
        else:
            last_target = df_district.iloc[-1][target]
        prev_lag = last_target
        for i in range(1, forecast_months + 1):
            next_date = last_date + pd.DateOffset(months=i)
            month_val = next_date.month
            year_val = next_date.year
            
            # Build dummy dictionary from dummy_cols.
            district_dummy = {col: 0 for col in dummy_cols}
            district_col = "District_" + district
            if district_col in district_dummy:
                district_dummy[district_col] = 1
            
            # Build a DataFrame for prediction with the correct feature order
            X_new = pd.DataFrame({
                'Lag_1': [prev_lag],
                'Month': [month_val],
                'Year': [year_val],
            })
            for col, val in district_dummy.items():
                X_new[col] = val
            X_new = X_new[feature_cols]
            
            # Generate prediction
            pred = model.predict(X_new)[0]
            all_future_predictions.append({
                'District': district,
                'Date': next_date,
                'Predicted': pred
            })
            
            # Update lag for next iteration
            prev_lag = pred
    return pd.DataFrame(all_future_predictions)

########################################
# Streamlit App Layout
########################################

st.title("Malaria Outbreak Prediction Web App")
st.markdown("### Predict and Visualize Malaria Trends Based on API Data")

# Load the processed data and pre-trained model
df_model, feature_cols, target, dummy_cols = get_processed_data()
model = load_model()

if df_model is None:
    st.error("Error fetching or processing data.")
    st.stop()

# Sidebar – User Input Options
st.sidebar.header("Prediction Settings")
# Use multiselect to allow forecasting for one or more districts (default: all)
district_list = sorted(df_model['District_'].unique())
selected_districts = st.sidebar.multiselect("Select District(s)", district_list, default=district_list)
forecast_months = st.sidebar.slider("Forecast Horizon (Months)", min_value=1, max_value=12, value=6)

# Get the last known target value for a district (only applicable if one district is selected)
if len(selected_districts) == 1:
    df_district = df_model[df_model['District_'] == selected_districts[0]].sort_values('Date_')
    if not df_district.empty:
        last_known_value = df_district.iloc[-1][target]
    else:
        last_known_value = 0
else:
    last_known_value = 0

current_override = st.sidebar.number_input("Override current month's cases (optional)", value=float(last_known_value), step=1.0)

# Button to trigger forecasting
if st.sidebar.button("Run Forecast"):
    with st.spinner("Generating forecast..."):
        df_future = forecast_future(model, df_model, feature_cols, target, dummy_cols,
                                    selected_districts, forecast_months, current_override)
        if df_future is not None and not df_future.empty:
            st.success("Forecast generated!")
            
            st.subheader("Future Predictions")
            st.dataframe(df_future)
            
            #############################
            # Visualization Section
            #############################
            
            # If only one district is selected, show detailed historical and forecast trends
            if len(selected_districts) == 1:
                st.subheader("Historical Trend")
                df_hist = df_model[df_model['District_'] == selected_districts[0]][['Date_', target]].copy()
                fig_hist = px.line(df_hist, x='Date_', y=target,
                                   title=f"Historical {target} in {selected_districts[0]}",
                                   markers=True)
                st.plotly_chart(fig_hist, use_container_width=True)
                
                st.subheader("Forecast Trend")
                fig_forecast = px.line(df_future[df_future['District'] == selected_districts[0]], x='Date', y='Predicted',
                                       title=f"Forecasted {target} for {selected_districts[0]}",
                                       markers=True)
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                st.subheader("Combined Historical & Forecast Trend")
                df_hist_renamed = df_hist.rename(columns={'Date_': 'Date', target: 'Cases'})
                df_future_renamed = df_future[df_future['District'] == selected_districts[0]].rename(columns={'Predicted': 'Cases'})
                df_combined = pd.concat([df_hist_renamed, df_future_renamed])
                fig_combined = px.line(df_combined, x='Date', y='Cases',
                                       title=f"Combined Historical and Forecasted {target} in {selected_districts[0]}",
                                       markers=True)
                st.plotly_chart(fig_combined, use_container_width=True)
                
                st.subheader("Monthly Cases Bar Chart")
                df_bar = df_hist.copy()
                df_bar['Month'] = df_bar['Date_'].dt.strftime('%b %Y')
                fig_bar = px.bar(df_bar, x='Month', y=target,
                                 title=f"Monthly {target} in {selected_districts[0]}")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Multiple districts selected. Showing combined forecast plot.")
            
            # Combined Forecast Plot for all selected districts
            st.subheader("Forecast Trend by District")
            fig_combined_all = px.line(df_future, x='Date', y='Predicted', color='District',
                                       title="Forecast of Future Malaria Predictions by District",
                                       markers=True)
            st.plotly_chart(fig_combined_all, use_container_width=True)
            
            # Heatmap: Cases by District & Month (All Districts)
            st.subheader("Heatmap: Cases by District & Month")
            df_heat = df_model.copy()
            df_heat['MonthYear'] = df_heat['Date_'].dt.strftime('%b %Y')
            heat_data = df_heat.pivot_table(index='District_', columns='MonthYear',
                                            values=target, aggfunc='sum', fill_value=0)
            fig_heat = px.imshow(heat_data,
                                 title=f"Heatmap of {target} Across Districts and Months",
                                 labels=dict(x="Month-Year", y="District", color=target))
            st.plotly_chart(fig_heat, use_container_width=True)
            
            # Correlation Heatmap of Engineered Features
            st.subheader("Correlation Heatmap of Features")
            corr_features = ['Lag_1', 'Month', 'Year'] + dummy_cols
            corr = df_model[corr_features + [target]].corr()
            fig_corr = px.imshow(corr, text_auto=True,
                                 title="Correlation Heatmap of Engineered Features and Target")
            st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("Adjust settings from the sidebar and click 'Run Forecast' to generate predictions.")
