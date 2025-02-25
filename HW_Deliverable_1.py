# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 14:27:24 2025

@author: jingm
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go

# 1. Read and clean data 
df_2009_2010 = pd.read_excel("./Spring 2025/Forecasting/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df_2010_2011 = pd.read_excel("./Spring 2025/Forecasting/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = pd.concat([df_2009_2010, df_2010_2011], ignore_index=True)

df.dropna(subset=['InvoiceDate', 'Quantity', 'Price'], inplace=True)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
df.dropna(subset=['InvoiceDate'], inplace=True)
df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
df['sales'] = df['Quantity'] * df['Price']
df.set_index('InvoiceDate', inplace=True)

# 2. Aggregate Sales by Month
monthly_sales = df.resample('M').sum(numeric_only=True)[['sales']]

# 3. STL Decomposition with Monthly Seasonality
stl = STL(monthly_sales['sales'], period=12, robust=True)  
result = stl.fit()
trend, seasonal, residual = result.trend, result.seasonal, result.resid

# 4. Train/Test Split
train_size = int(len(trend) * 0.8)
train_trend, test_trend = trend[:train_size], trend[train_size:]
train_seasonal, test_seasonal = seasonal[:train_size], seasonal[train_size:]
train_actual, test_actual = monthly_sales[:train_size], monthly_sales[train_size:]

# 5. Fit Holt-Winters Model on Trend Component
hw_model = ExponentialSmoothing(
    train_trend, 
    trend='add', 
    damped_trend=True
).fit(optimized=True)

# 6. Forecast Trend Component
forecast_horizon = len(test_trend) + 12  
trend_forecast = hw_model.forecast(steps=forecast_horizon)

# 7. Use SARIMA for Seasonal Component
sarima_model = SARIMAX(train_seasonal, order=(1, 0, 1), seasonal_order=(1, 1, 1, 12)).fit()
seasonal_forecast = sarima_model.forecast(steps=len(test_seasonal))

# 8. Combine Forecasted Trend + Seasonal
final_forecast = trend_forecast[:len(test_seasonal)] + seasonal_forecast

# 9. Plot Actual vs Forecasted Sales
##Matplotlib
plt.figure(figsize=(12, 6))
plt.plot(train_actual.index, train_actual['sales'], label="Actual Training Data", color="blue")
plt.plot(test_actual.index, test_actual['sales'], label="Actual Test Data", color="green")
plt.plot(test_trend.index, final_forecast, label="Final Forecast", color="red", linestyle="dashed")
plt.legend()
plt.title("STL + Holt-Winters for Trend + SARIMA for Monthly Seasonality")
plt.xlabel("Date")
plt.ylabel("Monthly Sales")
plt.show()
##Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=train_actual.index, y=train_actual['sales'], mode='lines', name='Actual Training Data', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=test_actual.index, y=test_actual['sales'], mode='lines', name='Actual Test Data', line=dict(color='green')))
fig.add_trace(go.Scatter(x=test_trend.index, y=final_forecast, mode='lines', name='Final Forecast', line=dict(color='red', dash='dash')))
fig.update_layout(
    title="STL + Holt-Winters for Trend + SARIMA for Seasonality",
    xaxis_title="Date",
    yaxis_title="Monthly Sales",
    legend_title="Legend",
)
fig.write_html("./Spring 2025/Forecasting/HW_forecast_plot.html")

# 10. Evaluate Model Performance
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mape = mean_absolute_percentage_error(test_trend, final_forecast)
