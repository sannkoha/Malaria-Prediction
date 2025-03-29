import numpy as np
import pandas as pd
import requests
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import joblib
import math
#Holt’s linear model:
# from statsmodels.tsa.holtwinters import Holt

# -------------------------------
# 1. API Data Fetching and Loading
# -------------------------------
url = "https://slhubmlapi.npha.gov.sl/External/pullMLData"
payload = {
    "_token": "620FF4EB-829F-4179-85F3-179524065C4E",
    "StartDate": "2025-01-14",  # expanded start date
    "EndDate": "2025-03-28"
}

response = requests.post(url, json=payload)
api_response = response.json()

if api_response.get('status') != 200:
    raise Exception("API Error: Status code is not 200.")

# Create DataFrame from API data
df = pd.DataFrame(api_response.get('data'))
print("API Data Shape:", df.shape)
print(df.head())

# -------------------------------
# 2. Exploratory Data Analysis (EDA)
# -------------------------------
plt.figure(figsize=(10,6))
sns.histplot(df['DHIS2AggregateValue'], bins=20, kde=True)
plt.title("Distribution of DHIS2AggregateValue")
plt.xlabel("Aggregate Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
sns.countplot(data=df, x='District', order=df['District'].value_counts().index)
plt.title("Count of Records by District")
plt.xlabel("District")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(data=df, x='TargetedDisease', order=df['TargetedDisease'].value_counts().index)
plt.title("Count of Records by Disease")
plt.xlabel("Disease")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Unique Diseases:", df['TargetedDisease'].unique())

# -------------------------------
# 3. Data Preprocessing and Cleaning
# -------------------------------
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

# -------------------------------
# 4. Additional Visualizations (Post Preprocessing)
# -------------------------------
overall_trend = df.groupby('Date')['DHIS2AggregateValue'].sum().reset_index()
plt.figure(figsize=(12,6))
plt.plot(overall_trend['Date'], overall_trend['DHIS2AggregateValue'], marker='o')
plt.title("Overall Time Trend of DHIS2AggregateValue")
plt.xlabel("Date")
plt.ylabel("Total Aggregate Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

disease_trend = df.groupby(['District', 'TargetedDisease', 'Date'])['DHIS2AggregateValue'].sum().reset_index()
plt.figure(figsize=(14,8))
sns.lineplot(data=disease_trend, x='Date', y='DHIS2AggregateValue', hue='TargetedDisease', style='District', markers=True)
plt.title("Disease Trends per District Over Time")
plt.xlabel("Date")
plt.ylabel("Aggregate Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

agg_district = df.groupby('District')['DHIS2AggregateValue'].sum().reset_index()
plt.figure(figsize=(14,7))
sns.barplot(x='District', y='DHIS2AggregateValue', data=agg_district, palette='viridis')
plt.title("Total Aggregate Value by District")
plt.xlabel("District")
plt.ylabel("Aggregate Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

agg_disease = df.groupby('TargetedDisease')['DHIS2AggregateValue'].sum().reset_index().sort_values(by='DHIS2AggregateValue', ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x='TargetedDisease', y='DHIS2AggregateValue', data=agg_disease, palette='rocket')
plt.title("Dominating Disease by Total Aggregate Value")
plt.xlabel("Disease")
plt.ylabel("Total Aggregate Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# 5. Data Aggregation and Feature Engineering for Forecasting
# -------------------------------
df_agg = df.groupby(['TargetedDisease', 'District', 'Date'])['DHIS2AggregateValue'].sum().reset_index()
df_agg['Month'] = df_agg['Date'].dt.month
df_agg['Year'] = df_agg['Date'].dt.year
df_agg['DateOrdinal'] = df_agg['Date'].apply(lambda x: x.toordinal())

# Create a time_index (months elapsed since first date)
start_date = df_agg['Date'].min()
df_agg['time_index'] = df_agg['Date'].apply(lambda x: (x.year - start_date.year) * 12 + (x.month - start_date.month))

# Add cyclical features for month
df_agg['MonthSin'] = df_agg['Month'].apply(lambda x: math.sin(2 * math.pi * x / 12))
df_agg['MonthCos'] = df_agg['Month'].apply(lambda x: math.cos(2 * math.pi * x / 12))

# One-hot encode the District variable.
df_model = pd.get_dummies(df_agg, columns=['District'], prefix='District')
print("Model Data Shape:", df_model.shape)
print(df_model.head())

# -------------------------------
# 6. Model Training per Disease (with Candidate Models)
# -------------------------------
disease_models = {}
diseases = df_model['TargetedDisease'].unique()
print("Unique Diseases for Modeling:", diseases)

for disease in diseases:
    print(f"\nProcessing disease: {disease}")
    df_disease = df_model[df_model['TargetedDisease'] == disease].copy()
    df_disease.sort_values(by='Date', inplace=True)
    
    # Feature set: use time_index, DateOrdinal, cyclical features, and district dummies.
    feature_cols = ['time_index', 'DateOrdinal', 'MonthSin', 'MonthCos'] + \
                   [col for col in df_disease.columns if col.startswith("District_")]
    target_col = 'DHIS2AggregateValue'
    
    # Time-based split: first 80% for training, last 20% for testing.
    split_index = int(len(df_disease) * 0.8)
    X_train = df_disease[feature_cols].iloc[:split_index]
    y_train = df_disease[target_col].iloc[:split_index]
    X_test = df_disease[feature_cols].iloc[split_index:]
    y_test = df_disease[target_col].iloc[split_index:]
    
    print("Training samples:", X_train.shape[0], "Testing samples:", X_test.shape[0])
    
    # Candidate Model 1: Random Forest
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    pred_rf = model_rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
    
    # Candidate Model 2: Gradient Boosting
    model_gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_gb.fit(X_train, y_train)
    pred_gb = model_gb.predict(X_test)
    rmse_gb = np.sqrt(mean_squared_error(y_test, pred_gb))
    
    # Candidate Model 3: Polynomial Regression (degree=2)
    model_poly = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model_poly.fit(X_train, y_train)
    pred_poly = model_poly.predict(X_test)
    rmse_poly = np.sqrt(mean_squared_error(y_test, pred_poly))
    
    print(f"RMSE for RandomForest: {rmse_rf:.2f}")
    print(f"RMSE for GradientBoosting: {rmse_gb:.2f}")
    print(f"RMSE for PolyRegression: {rmse_poly:.2f}")
    
    # Select the best model (lowest RMSE)
    candidate_models = {
        "RandomForest": (model_rf, rmse_rf),
        "GradientBoosting": (model_gb, rmse_gb),
        "PolyRegression": (model_poly, rmse_poly)
    }
    best_model_name = min(candidate_models, key=lambda k: candidate_models[k][1])
    best_model = candidate_models[best_model_name][0]
    best_rmse = candidate_models[best_model_name][1]
    print(f"Selected model for {disease}: {best_model_name} (RMSE: {best_rmse:.2f})")
    
    # Calculate residual standard deviation on training set for noise injection.
    residual_std = np.std(y_train - best_model.predict(X_train))
    
    # Save the best model and residual_std.
    model_filename = f"best_model_{disease.lower().replace(' ', '_')}.pkl"
    joblib.dump(best_model, model_filename)
    
    disease_models[disease] = {
        "model": best_model,
        "features": feature_cols,
        "target": target_col,
        "data": df_disease,
        "residual_std": residual_std  # for noise injection in forecasting
    }
    
    # Optionally, a Holt's linear trend model if data permits:
    # if len(df_disease) >= 3:
    #     series = df_disease.set_index('Date')[target_col]
    #     holt_model = Holt(series).fit(optimized=True)
    #     holt_rmse = np.sqrt(mean_squared_error(y_test, holt_model.fittedvalues[-len(y_test):]))
    #     print(f"Holt's RMSE for {disease}: {holt_rmse:.2f}")

# -------------------------------
# 7. Forecasting Functions (with noise injection)
# -------------------------------
def forecast_disease(disease, district, future_months=6, noise_injection=True):
    """
    Forecast future aggregate values for a given disease and district over future_months.
    If the district is not present in historical data, the district dummies remain 0.
    Optionally, random noise (based on training residuals) is added to enhance forecast variety.
    """
    if disease not in disease_models:
        raise Exception(f"No model found for disease: {disease}")
    
    model_info = disease_models[disease]
    model = model_info["model"]
    features = model_info["features"]
    df_disease = model_info["data"]
    residual_std = model_info["residual_std"]
    
    last_date = df_disease['Date'].max()
    last_time_index = df_disease.iloc[-1]['time_index']
    
    forecasts = []
    for i in range(1, future_months + 1):
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
        # Set district one-hot dummies.
        district_cols = [col for col in features if col.startswith("District_")]
        for col in district_cols:
            record[col] = 0
        district_dummy = "District_" + district
        if district_dummy in district_cols:
            record[district_dummy] = 1
        
        X_new = pd.DataFrame([record])[features]
        pred = model.predict(X_new)[0]
        
        # Inject random noise based on residual_std to increase variety.
        if noise_injection:
            noise = np.random.normal(scale=residual_std)
            pred += noise
        
        forecasts.append({
            'Disease': disease,
            'District': district,
            'Date': next_date,
            'PredictedAggregateValue': pred
        })
    
    return pd.DataFrame(forecasts)

# -------------------------------
# 8. Forecasting and Plotting
# -------------------------------
district_forecast = forecast_disease("Total Malaria Tested", "Bo District", future_months=6)
print("District-level Forecast for Total Malaria Tested in Bo District:")
print(district_forecast)

plt.figure(figsize=(12,8))
sns.lineplot(data=district_forecast, x='Date', y='PredictedAggregateValue', marker='o')
plt.title("District-Level Forecast of Total Malaria Tested (Bo District)")
plt.xlabel("Date")
plt.ylabel("Predicted Aggregate Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


def forecast_country_level(disease, future_months=6):
    return forecast_disease(disease, district="CountryWide", future_months=future_months)

country_forecast = forecast_country_level("Total Malaria Tested", future_months=6)
print("Country-level Forecast for Total Malaria Tested:")
print(country_forecast)

plt.figure(figsize=(12,8))
sns.lineplot(data=country_forecast, x='Date', y='PredictedAggregateValue', marker='o')
plt.title("Country-Level Forecast of Total Malaria Tested")
plt.xlabel("Date")
plt.ylabel("Predicted Aggregate Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
