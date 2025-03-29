import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
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
    # Use last available date and time_index from training data
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
        
        # Build a record with all features used in training
        record = {
            'time_index': time_index,
            'DateOrdinal': date_ordinal,
            'MonthSin': month_sin,
            'MonthCos': month_cos
        }
        # Initialize all district dummy features to 0
        for col in feature_cols:
            if col.startswith("District_"):
                record[col] = 0
        # Set the dummy for the chosen district (if exists)
        district_dummy = "District_" + district
        if district_dummy in feature_cols:
            record[district_dummy] = 1
        
        X_new = pd.DataFrame([record])[feature_cols]
        pred = model.predict(X_new)[0]
        # Optional: add random noise and override adjustment
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

available_diseases = sorted(df_model['Disease'].unique())
selected_disease = st.sidebar.selectbox("Select Disease", available_diseases)

# For district selection, filter the one-hot encoded data by disease
disease_df = df_model[df_model['Disease'] == selected_disease]
# Get unique district names from the original df_agg
available_districts = sorted(df_agg[df_agg['Disease'] == selected_disease]['District'].unique())
selected_districts = st.sidebar.multiselect("Select District(s)", available_districts, default=available_districts[:1])

forecast_months = st.sidebar.slider("Forecast Horizon (Months)", 1, 12, 6)
override_patients = st.sidebar.number_input("Override Patients (adjust forecast)", value=0)

# Option for country-level forecast (ignores district selection)
country_forecast_option = st.sidebar.checkbox("Generate Country-Level Forecast", value=False)

# Run Forecast Button
if st.sidebar.button("Run Forecast"):
    st.sidebar.info("Running forecast, please wait...")
    
    # Load saved model for the selected disease
    model = load_model(selected_disease)
    if model is None:
        st.error("Model loading failed. Please check the model file.")
        st.stop()
    
    # Retrieve feature columns exactly as in training.
    dummy_cols = [col for col in df_model.columns if col.startswith("District_")]
    feature_cols = ['time_index', 'DateOrdinal', 'MonthSin', 'MonthCos'] + dummy_cols
    
    # Get training data for the selected disease from the one-hot encoded dataframe
    df_disease = df_model[df_model['Disease'] == selected_disease].copy()
    df_disease.sort_values(by='Date', inplace=True)
    
    # For this demo, set a default residual_std (in production, load stored metadata)
    residual_std = 0.0
    
    # Generate forecast for selected districts if not country-level;
    # otherwise, forecasts_all is empty (we'll later generate and display country-level separately)
    forecasts_all = pd.DataFrame()
    if not country_forecast_option:
        for dist in selected_districts:
            df_forecast = forecast_disease(model, df_disease, feature_cols, forecast_months,
                                           override=override_patients, residual_std=residual_std,
                                           district=dist)
            forecasts_all = pd.concat([forecasts_all, df_forecast], axis=0)
    
    if not forecasts_all.empty:
        st.success("District-level forecast generated!")
        st.subheader("Future Predictions (District-level)")
        st.dataframe(forecasts_all)
        
        # Visualization Section using Plotly
        import plotly.express as px
        
        if len(selected_districts) == 1:
            # Historical Trend for the selected district (from original aggregated data)
            hist_df = df_agg[(df_agg['Disease'] == selected_disease) & (df_agg['District'] == selected_districts[0])]
            st.subheader(f"Historical Trend in {selected_districts[0]}")
            fig_hist = px.line(hist_df, x="Date", y="DHIS2AggregateValue",
                               title=f"Historical {selected_disease} Cases in {selected_districts[0]}",
                               markers=True)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Forecast Trend for the selected district
            st.subheader(f"Forecast Trend in {selected_districts[0]}")
            fig_forecast = px.line(forecasts_all[forecasts_all['District'] == selected_districts[0]],
                                   x="Date", y="Predicted",
                                   title=f"Forecasted {selected_disease} Cases in {selected_districts[0]}",
                                   markers=True)
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Combined Historical & Forecast Trend
            st.subheader("Combined Historical & Forecast Trend")
            hist_df_renamed = hist_df.rename(columns={"DHIS2AggregateValue": "Cases"})
            forecast_df_renamed = forecasts_all[forecasts_all['District'] == selected_districts[0]].rename(columns={"Predicted": "Cases"})
            combined_df = pd.concat([hist_df_renamed[['Date','Cases']], forecast_df_renamed[['Date','Cases']]])
            fig_combined = px.line(combined_df, x="Date", y="Cases",
                                   title=f"Combined Trend in {selected_districts[0]}",
                                   markers=True)
            st.plotly_chart(fig_combined, use_container_width=True)
            
            # Monthly Cases Bar Chart
            st.subheader("Monthly Cases Bar Chart")
            hist_df['MonthYear'] = hist_df['Date'].dt.strftime('%b %Y')
            fig_bar = px.bar(hist_df, x="MonthYear", y="DHIS2AggregateValue",
                             title=f"Monthly Cases in {selected_districts[0]}")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Multiple districts selected. Showing combined forecast plot.")
        
        # Combined Forecast Plot for all selected districts
        st.subheader("Forecast Trend by District")
        fig_combined_all = px.line(forecasts_all, x="Date", y="Predicted", color="District",
                                   title="Forecast of Future Predictions by District",
                                   markers=True)
        st.plotly_chart(fig_combined_all, use_container_width=True)
        
        # Heatmap: Cases by District & Month (All Districts)
        st.subheader("Heatmap: Cases by District & Month")
        heat_df = df_agg[df_agg['Disease'] == selected_disease].copy()
        heat_df['MonthYear'] = heat_df['Date'].dt.strftime('%b %Y')
        heat_data = heat_df.pivot_table(index='District', columns='MonthYear', values='DHIS2AggregateValue', aggfunc='sum', fill_value=0)
        fig_heat = px.imshow(heat_data, text_auto=True,
                             title=f"Heatmap of {selected_disease} Cases Across Districts and Months",
                             labels=dict(x="Month-Year", y="District", color="Cases"))
        st.plotly_chart(fig_heat, use_container_width=True)
    
    # Separate country-level forecast visualization at the end
    if country_forecast_option:
        st.success("Country-level forecast generated!")
        # Generate country-level forecast using district dummy "CountryWide"
        country_forecast = forecast_disease(model, df_disease, feature_cols, forecast_months,
                                            override=override_patients, residual_std=residual_std,
                                            district="CountryWide")
        st.subheader("Future Predictions (Country-level)")
        st.dataframe(country_forecast)
        
        st.subheader("Country-Level Forecast Trend")
        fig_country = px.line(country_forecast, x="Date", y="Predicted",
                              title=f"Country-Level Forecast of {selected_disease} Cases",
                              markers=True)
        st.plotly_chart(fig_country, use_container_width=True)
else:
    st.info("Adjust settings from the sidebar and click 'Run Forecast' to generate predictions.")
