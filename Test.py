import numpy as np
import pandas as pd
import requests
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import joblib


# -------------------------------
# 1. API Data Fetching and Loading
# -------------------------------
# Expand the date range to fetch more historical data
url = "http://hubapi.duchimatech.com/External/pullMLData"
payload = {
    "_token": "F1A7D842-36B5-4C8C-A2B5-6F9D19D2B073",
    "StartDate": "2023-01-01",  # expanded start date
    "EndDate": "2025-02-28"
}

response = requests.post(url, json=payload)
api_response = response.json()

if api_response.get('status') != 200:
    raise Exception("API Error: Status code is not 200.")

# Create DataFrame from API data
df_api = pd.DataFrame(api_response.get('data'))
print("API Data Shape:", df_api.shape)
print(df_api.head())



# -------------------------------
# 1A. Additional Visualizations on Raw API Data
# -------------------------------
# Histogram of Aggregate Values
plt.figure(figsize=(10,6))
sns.histplot(df_api['DHIS2AggregateValue'], bins=20, kde=True)
plt.title("Distribution of DHIS2AggregateValue (Raw API Data)")
plt.xlabel("Aggregate Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Count plot of records by District (or FacilityName if District is missing)
plt.figure(figsize=(12,6))
sns.countplot(data=df_api, x='District', order=df_api['District'].value_counts().index)
plt.title("Count of Records by District")
plt.xlabel("District")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# -------------------------------
# 2. Data Preprocessing and Cleaning
# -------------------------------
# Convert DHIS2Period (e.g., "202412") into a proper Date column (set to first day of month)
def parse_dhis2period(period_str):
    try:
        year = int(period_str[:4])
        month = int(period_str[4:])
        return pd.Timestamp(year=year, month=month, day=1)
    except Exception as e:
        return pd.NaT

df_api['Date'] = df_api['DHIS2Period'].apply(parse_dhis2period)

# For identifier, if 'District' is missing then use 'FacilityName'
df_api['District'] = df_api['District'].fillna(df_api['FacilityName'])

# Create a function to extract MeasurementType and AgeGroup from DataElementName.
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

print(df_api[['DataElementName', 'MeasurementType', 'AgeGroup', 'Date']].head())

# -------------------------------
# 3. Reshape Data for Analysis
# -------------------------------
# Pivot the data so each row is identified by District and Date,
# with columns for each combination of MeasurementType and AgeGroup.
df_pivot = df_api.pivot_table(
    index=['District', 'Date'],
    columns=['MeasurementType', 'AgeGroup'],
    values='DHIS2AggregateValue',
    aggfunc='sum',
    fill_value=0
).reset_index()

# Flatten the MultiIndex columns
df_pivot.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_pivot.columns.values]
print("Pivoted DataFrame Head:")
print(df_pivot.head())

# Additional visualization of pivoted data:
# Heatmap to show the distribution of victims (if available) per district over time
if 'Victims_gt5' in df_pivot.columns or 'Victims_lt5' in df_pivot.columns:
    # Melt data for a heatmap (using Date vs District for one of the victim metrics)
    if 'Victims_gt5' in df_pivot.columns:
        heatmap_data = df_pivot.pivot(index='District_', columns='Date_', values='Victims_gt5')
    else:
        heatmap_data = df_pivot.pivot(index='District_', columns='Date_', values='Victims_lt5')
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".0f")
    plt.title("Heatmap of Victims (gt5 or lt5) by District and Date")
    plt.xlabel("Date")
    plt.ylabel("District")
    plt.tight_layout()
    plt.show()

# -------------------------------
# 4. Visualization: Trends and Comparisons (Post Preprocessing)
# -------------------------------
# (a) Overall Trend: Aggregate values across all Districts by Date
trend_df = df_api.groupby('Date')['DHIS2AggregateValue'].sum().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(trend_df['Date'], trend_df['DHIS2AggregateValue'], marker='o')
plt.title('Monthly Malaria Aggregate Values Trend (API Data)')
plt.xlabel('Date')
plt.ylabel('Aggregate Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# (b) If multiple age groups exist (e.g., lt5 and gt5 for the same measurement), plot them separately.
if 'Victims_gt5' in df_pivot.columns or 'Victims_lt5' in df_pivot.columns:
    age_cols = [col for col in df_pivot.columns if col.startswith('Victims')]
    df_melt = df_pivot.melt(id_vars=['District_', 'Date_'], value_vars=age_cols,
                            var_name='Metric', value_name='Count')
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df_melt, x='Date_', y='Count', hue='Metric', marker='o')
    plt.title('Monthly Malaria Victims by Age Group')
    plt.xlabel('Date')
    plt.ylabel('Aggregate Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# (c) District-wise Comparison (if you have multiple facilities/districts)
agg_district = df_api.groupby('District')['DHIS2AggregateValue'].sum().reset_index()
plt.figure(figsize=(14, 7))
sns.barplot(x='District', y='DHIS2AggregateValue', data=agg_district, palette='viridis')
plt.title('Total Aggregate Values by District')
plt.xlabel('District')
plt.ylabel('Aggregate Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# 5. Feature Engineering for Modeling
# -------------------------------
# For our modeling, we choose a target variable.
# If both lt5 and gt5 exist, combine them.
if 'Victims_gt5' in df_pivot.columns and 'Victims_lt5' in df_pivot.columns:
    df_pivot['Total_Victims'] = df_pivot['Victims_lt5'] + df_pivot['Victims_gt5']
    target = 'Total_Victims'
elif any(col.startswith('Victims') for col in df_pivot.columns):
    target = [col for col in df_pivot.columns if col.startswith('Victims')][0]
else:
    target = 'DHIS2AggregateValue'  # fallback

# Sort by District and Date
df_pivot.sort_values(by=['District_', 'Date_'], inplace=True)

# Create additional lag features to capture temporal dependencies
df_pivot['Lag_1'] = df_pivot.groupby('District_')[target].shift(1)
df_pivot['Lag_2'] = df_pivot.groupby('District_')[target].shift(2)
df_pivot['Lag_3'] = df_pivot.groupby('District_')[target].shift(3)

# Create a rolling mean feature (e.g., over the last 3 months)
df_pivot['Rolling_Mean_3'] = df_pivot.groupby('District_')[target].transform(lambda x: x.rolling(3, min_periods=1).mean())

# Add time features (month and year)
df_pivot['Month'] = df_pivot['Date_'].dt.month
df_pivot['Year'] = df_pivot['Date_'].dt.year

# One-hot encode the District (if more than one exists) so that the model can learn district-specific trends.
district_dummies = pd.get_dummies(df_pivot['District_'], prefix='District_')
df_model = pd.concat([df_pivot, district_dummies], axis=1)

# Drop rows with missing lag values (first few observations per District)
df_model = df_model.dropna(subset=['Lag_1', 'Lag_2', 'Lag_3']).reset_index(drop=True)
print("Modeling Data Shape:", df_model.shape)
print(df_model[['District_', 'Date_', target, 'Lag_1', 'Lag_2', 'Lag_3', 'Rolling_Mean_3', 'Month', 'Year']].head())

# Additional Visualization: Correlation heatmap of engineered features and target
features = ['Lag_1', 'Lag_2', 'Lag_3', 'Rolling_Mean_3', 'Month', 'Year'] + list(district_dummies.columns)
plt.figure(figsize=(10,8))
corr = df_model[features + [target]].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Engineered Features and Target")
plt.tight_layout()
plt.show()

# -------------------------------
# 6. Train-Test Split and Modeling
# -------------------------------
# Define features and target
X = df_model[features]
y = df_model[target]

# Simple time-based split: 80% training and 20% testing
split_index = int(len(df_model) * 0.8)
X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]
X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

print("Training Features Shape:", X_train.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Testing Target Shape:", y_test.shape)

# Initialize and train the Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on test set and evaluate
y_pred_rf = rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
print("Random Forest RMSE:", rmse_rf)
print("Random Forest MAE:", mae_rf)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=2,  
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)
best_rf_model = grid_search.best_estimator_

# Evaluate the best model
y_train_pred_best = best_rf_model.predict(X_train)
y_test_pred_best = best_rf_model.predict(X_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred_best))
mae_train = mean_absolute_error(y_train, y_train_pred_best)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred_best))
mae_test = mean_absolute_error(y_test, y_test_pred_best)

print("Training Set Evaluation:")
print("Best RF RMSE:", rmse_train, "MAE:", mae_train)
print("Testing Set Evaluation:")
print("Best RF RMSE:", rmse_test, "MAE:", mae_test)


# Save the best model for forecasting
joblib.dump(best_rf_model, 'best_rf_model_api_improved.pkl')

# -------------------------------
# 7. Forecasting Future Malaria Trends using the Saved Model
# -------------------------------
# Load the saved model
model_rf = joblib.load('best_rf_model_api_improved.pkl')

# Get the last known data point from your historical data
last_date = df_model['Date_'].max()
last_target = df_model.sort_values('Date_').tail(1)[target].values[0]
district = df_model.sort_values('Date_').tail(1)['District_'].values[0]

# Define how many future months to forecast
future_months = 6
future_predictions = []

# For forecasting, we need to update our lag features iteratively.
# We'll initialize our lag values with the last observed values.
prev_lag1 = last_target
prev_lag2 = df_model.sort_values('Date_').tail(1)['Lag_1'].values[0]
prev_lag3 = df_model.sort_values('Date_').tail(1)['Lag_2'].values[0]

for i in range(1, future_months + 1):
    next_date = last_date + pd.DateOffset(months=i)
    month_val = next_date.month
    year_val = next_date.year
    
    # One-hot encode the current district for prediction
    district_dummy = {col: 0 for col in district_dummies.columns}
    district_col = "District_" + district
    if district_col in district_dummy:
        district_dummy[district_col] = 1

    # Create the rolling mean using the current lag values
    rolling_mean = np.mean([prev_lag1, prev_lag2, prev_lag3])
    
    # Build feature vector
    X_new = pd.DataFrame({
        'Lag_1': [prev_lag1],
        'Lag_2': [prev_lag2],
        'Lag_3': [prev_lag3],
        'Rolling_Mean_3': [rolling_mean],
        'Month': [month_val],
        'Year': [year_val],
    })
    for col, val in district_dummy.items():
        X_new[col] = val
    
    # Predict the next month's target
    pred = model_rf.predict(X_new)[0]
    
    future_predictions.append({
        'District': district,
        'Date': next_date,
        'Predicted': pred
    })
    
    # Update lag features for the next iteration:
    prev_lag3 = prev_lag2
    prev_lag2 = prev_lag1
    prev_lag1 = pred

# Convert future predictions to a DataFrame and visualize
df_future = pd.DataFrame(future_predictions)
print("Future Predictions:")
print(df_future)

plt.figure(figsize=(10, 6))
plt.plot(df_future['Date'], df_future['Predicted'], marker='o', linestyle='-')
plt.title('Forecast of Future Malaria Predictions')
plt.xlabel('Date')
plt.ylabel('Predicted Victims Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

