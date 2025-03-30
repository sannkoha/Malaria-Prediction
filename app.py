import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import joblib
from datetime import datetime
import math

# -------------------------------
# Helper Functions
# -------------------------------

@st.cache_data(show_spinner=False)
def fetch_api_data():
    url = "https://slhubmlapi.npha.gov.sl/External/pullMLData"
    payload = {
        "_token": "620FF4EB-829F-4179-85F3-179524065C4E",
        "StartDate": "2025-01-14",
        "EndDate": "2025-03-28"
    }
    response = requests.post(url, json=payload)
    api_response = response.json()
    if api_response.get('status') != 200:
        st.error("API Error: Status code is not 200.")
        return None
    df = pd.DataFrame(api_response.get('data'))
    # Convert DHIS2Period to Date
    def parse_dhis2period(period_str):
        try:
            if "W" in period_str:
                year = int(period_str[:4])
                week = int(period_str.split("W")[1])
                return pd.to_datetime(f'{year}-01-01') + pd.Timedelta(days=(week-1)*7)
            else:
                year = int(period_str[:4])
                month = int(period_str[4:])
                return pd.Timestamp(year=year, month=month, day=1)
        except Exception as e:
            return pd.NaT
    df['Date'] = df['DHIS2Period'].apply(parse_dhis2period)
    df['District'] = df['District'].fillna("Unknown District")
    return df

@st.cache_data(show_spinner=False)
def preprocess_data(df):
    # Aggregate data by Disease, District and Date
    df_agg = df.groupby(['TargetedDisease', 'District', 'Date'])['DHIS2AggregateValue'].sum().reset_index()
    df_agg['Month'] = df_agg['Date'].dt.month
    df_agg['Year'] = df_agg['Date'].dt.year
    df_agg['DateOrdinal'] = df_agg['Date'].apply(lambda x: x.toordinal())
    # Create a time_index (months elapsed since first date)
    start_date = df_agg['Date'].min()
    df_agg['time_index'] = df_agg['Date'].apply(lambda x: (x.year - start_date.year) * 12 + (x.month - start_date.month))
    # Add cyclical features for seasonality
    df_agg['MonthSin'] = df_agg['Month'].apply(lambda x: math.sin(2 * math.pi * x / 12))
    df_agg['MonthCos'] = df_agg['Month'].apply(lambda x: math.cos(2 * math.pi * x / 12))
    # Rename for consistency with training
    df_agg.rename(columns={'TargetedDisease': 'Disease'}, inplace=True)
    return df_agg

def one_hot_encode(df_agg):
    # Mimic training: apply one-hot encoding on 'District'
    df_model = pd.get_dummies(df_agg, columns=['District'], prefix='District')
    return df_model

def load_model(disease):
    # Assumes saved model filename follows the convention "best_model_<disease>.pkl"
    filename = f"best_model_{disease.lower().replace(' ', '_')}.pkl"
    try:
        model = joblib.load(filename)
    except Exception as e:
        st.error(f"Could not load model for {disease}: {e}")
        model = None
    return model

def forecast_disease(model, df_disease, feature_cols, forecast_months, override=0, residual_std=0.0, district="CountryWide"):
    """
    Generate forecasts using the loaded model.
      - df_disease: training data for the disease (after one-hot encoding)
      - feature_cols: list of feature columns used during training
      - forecast_months: number of future months to forecast
      - override: adjustment value to add to forecasted results
      - residual_std: used for optional noise injection
      - district: the district to forecast for
    """
    last_date = df_disease['Date'].max()
    last_time_index = df_disease.iloc[-1]['time_index']
    forecasts = []
    for i in range(1, forecast_months + 1):
        next_date = last_date + pd.DateOffset(months=i)
        time_index = last_time_index + i
        date_ordinal = next_date.toordinal()
        month = next_date.month
        month_sin = math.sin(2 * math.pi * month / 12)
        month_cos = math.cos(2 * math.pi * month / 12)
        record = {
            'time_index': time_index,
            'DateOrdinal': date_ordinal,
            'MonthSin': month_sin,
            'MonthCos': month_cos
        }
        # Set all district dummy features to 0
        for col in feature_cols:
            if col.startswith("District_"):
                record[col] = 0
        district_dummy = "District_" + district
        if district_dummy in feature_cols:
            record[district_dummy] = 1
        X_new = pd.DataFrame([record])[feature_cols]
        pred = model.predict(X_new)[0]
        noise = np.random.normal(scale=residual_std) if residual_std > 0 else 0
        pred = pred + noise + override
        forecasts.append({
            'Disease': df_disease.iloc[0]['Disease'],
            'District': district,
            'Date': next_date,
            'Predicted': pred
        })
    return pd.DataFrame(forecasts)

# -------------------------------
# Streamlit App Layout
# -------------------------------

st.set_page_config(page_title="Disease Outbreak Forecasting", layout="wide")
st.title("Disease Outbreak Forecasting App")

# Sidebar Options
st.sidebar.header("Forecast Settings")
df_api = fetch_api_data()
if df_api is None:
    st.stop()
df_agg = preprocess_data(df_api)
df_model = one_hot_encode(df_agg)

# Allow multiple disease selection
available_diseases = sorted(df_model['Disease'].unique())
selected_diseases = st.sidebar.multiselect("Select Disease(s)", available_diseases, default=available_diseases[:1])

# Global district selection (applied per disease)
# Note: District options may differ per disease; here we assume a common selection.
all_districts = sorted(df_agg['District'].unique())
selected_districts = st.sidebar.multiselect("Select District(s)", all_districts, default=all_districts[:1])

forecast_months = st.sidebar.slider("Forecast Horizon (Months)", 1, 12, 6)
override_patients = st.sidebar.number_input("Override Patients (adjust forecast)", value=0)
country_forecast_option = st.sidebar.checkbox("Generate Country-Level Forecast", value=False)

# Run Forecast Button
if st.sidebar.button("Run Forecast"):
    st.sidebar.info("Running forecast, please wait...")
    all_forecasts = pd.DataFrame()
    # Retrieve feature columns exactly as in training
    dummy_cols = [col for col in df_model.columns if col.startswith("District_")]
    feature_cols = ['time_index', 'DateOrdinal', 'MonthSin', 'MonthCos'] + dummy_cols

    # Loop over each selected disease
    for disease in selected_diseases:
        st.subheader(f"Forecast for {disease}")
        model = load_model(disease)
        if model is None:
            st.error(f"Model loading failed for {disease}. Skipping...")
            continue
        # Get training data for this disease
        df_disease = df_model[df_model['Disease'] == disease].copy()
        df_disease.sort_values(by='Date', inplace=True)
        # Set default residual_std (adjust or load stored metadata as needed)
        residual_std = 0.0

        # District-level Forecasts (if not country-level)
        if not country_forecast_option:
            disease_forecasts = pd.DataFrame()
            # For each selected district, filter by disease-specific available districts
            available_districts_disease = sorted(df_agg[df_agg['Disease'] == disease]['District'].unique())
            selected_districts_disease = [d for d in selected_districts if d in available_districts_disease]
            if not selected_districts_disease:
                st.info(f"No matching district found for {disease}. Skipping district-level forecast.")
            else:
                for dist in selected_districts_disease:
                    df_forecast = forecast_disease(model, df_disease, feature_cols, forecast_months,
                                                   override=override_patients, residual_std=residual_std,
                                                   district=dist)
                    disease_forecasts = pd.concat([disease_forecasts, df_forecast], axis=0)
                if not disease_forecasts.empty:
                    st.success(f"District-level forecast generated for {disease}!")
                    st.dataframe(disease_forecasts)
                    # Display visualizations for each disease
                    import plotly.express as px
                    if len(selected_districts_disease) == 1:
                        # Historical Trend for this disease & district
                        hist_df = df_agg[(df_agg['Disease'] == disease) & 
                                         (df_agg['District'] == selected_districts_disease[0])]
                        st.subheader(f"Historical Trend in {selected_districts_disease[0]} for {disease}")
                        fig_hist = px.line(hist_df, x="Date", y="DHIS2AggregateValue",
                                           title=f"Historical {disease} Cases in {selected_districts_disease[0]}",
                                           markers=True)
                        st.plotly_chart(fig_hist, use_container_width=True)
                        st.subheader(f"Forecast Trend in {selected_districts_disease[0]} for {disease}")
                        fig_forecast = px.line(disease_forecasts[disease_forecasts['District'] == selected_districts_disease[0]],
                                               x="Date", y="Predicted",
                                               title=f"Forecasted {disease} Cases in {selected_districts_disease[0]}",
                                               markers=True)
                        st.plotly_chart(fig_forecast, use_container_width=True)
                    else:
                        st.info(f"Multiple districts selected for {disease}. Showing combined forecast plot.")
                    st.subheader(f"Combined Forecast Trend by District for {disease}")
                    fig_combined = px.line(disease_forecasts, x="Date", y="Predicted", color="District",
                                           title=f"Forecast of Future Predictions for {disease}",
                                           markers=True)
                    st.plotly_chart(fig_combined, use_container_width=True)
                    st.subheader(f"Heatmap: {disease} Cases by District & Month")
                    heat_df = df_agg[df_agg['Disease'] == disease].copy()
                    heat_df['MonthYear'] = heat_df['Date'].dt.strftime('%b %Y')
                    heat_data = heat_df.pivot_table(index='District', columns='MonthYear', 
                                                    values='DHIS2AggregateValue', aggfunc='sum', fill_value=0)
                    fig_heat = px.imshow(heat_data, text_auto=True,
                                         title=f"Heatmap of {disease} Cases Across Districts and Months",
                                         labels=dict(x="Month-Year", y="District", color="Cases"))
                    st.plotly_chart(fig_heat, use_container_width=True)
        # Country-level Forecast Section (displayed separately below)
        if country_forecast_option:
            st.subheader(f"Country-Level Forecast for {disease}")
            country_forecast = forecast_disease(model, df_disease, feature_cols, forecast_months,
                                                override=override_patients, residual_std=residual_std,
                                                district="CountryWide")
            st.dataframe(country_forecast)
            fig_country = px.line(country_forecast, x="Date", y="Predicted",
                                  title=f"Country-Level Forecast of {disease} Cases",
                                  markers=True)
            st.plotly_chart(fig_country, use_container_width=True)
else:
    st.info("Adjust settings from the sidebar and click 'Run Forecast' to generate predictions.")
