"""
AI Stock Price Predictor — AAPL
================================
Fetches live stock data, engineers technical indicators, trains ML models
to predict next-day closing price, and exports everything to Excel for
Power BI dashboard connection.

Run this script anytime to refresh data with the latest market prices.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────
TICKER = "AAPL"
PERIOD = "3y"          # 3 years of historical data
INTERVAL = "1d"        # daily data

print(f"{'='*60}")
print(f"AI STOCK PRICE PREDICTOR — {TICKER}")
print(f"{'='*60}\n")

# ── 1. FETCH LIVE DATA ────────────────────────────────────────────────────
print(f"Fetching {PERIOD} of data for {TICKER}...")
df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)
df.reset_index(inplace=True)

# Flatten multi-index columns if present (newer yfinance versions)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [c[0] for c in df.columns]

print(f"Fetched {len(df)} rows | Date range: {df['Date'].min().date()} to {df['Date'].max().date()}\n")

# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────
print("Engineering technical indicators...")

# Moving Averages
df['MA7']  = df['Close'].rolling(window=7).mean()
df['MA21'] = df['Close'].rolling(window=21).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()

# Daily Return %
df['Daily Return'] = df['Close'].pct_change() * 100

# RSI (14-day Relative Strength Index)
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD (Moving Average Convergence Divergence)
ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Volume change %
df['Volume Change'] = df['Volume'].pct_change() * 100

# Lag features (previous days' closing prices)
df['Close Lag1'] = df['Close'].shift(1)
df['Close Lag2'] = df['Close'].shift(2)
df['Close Lag3'] = df['Close'].shift(3)

# Target: NEXT DAY's closing price
df['Target'] = df['Close'].shift(-1)

# Buy/Sell/Hold signal based on MA crossover (for dashboard context)
df['Signal'] = np.where(df['MA7'] > df['MA21'], 'Buy', 'Sell')

# Save full dataset with indicators (before dropping NaNs) for charting
df_full = df.copy()

# Drop rows with NaN (from rolling windows / shift)
df_model = df.dropna().reset_index(drop=True)
print(f"Usable rows after feature engineering: {len(df_model)}\n")

# ── 3. TRAIN MODELS ────────────────────────────────────────────────────────
print("Training models...")

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA7', 'MA21', 'MA50',
             'Daily Return', 'RSI', 'MACD', 'Signal Line', 'Volume Change',
             'Close Lag1', 'Close Lag2', 'Close Lag3']

X = df_model[features]
y = df_model['Target']

# Time-based split (80% train, 20% test) — NEVER shuffle time series data
split_idx = int(len(df_model) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_test = df_model['Date'].iloc[split_idx:].reset_index(drop=True)

# Scale features for Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

# Model 2: Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# ── 4. EVALUATE ────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R2 Score: {r2:.4f}")
    return {"Model": name, "RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2 Score": round(r2, 4)}

lr_metrics = evaluate(y_test, lr_pred, "Linear Regression")
rf_metrics = evaluate(y_test, rf_pred, "Random Forest")

# ── 5. PREDICT TOMORROW'S PRICE ─────────────────────────────────────────────
latest_features = X.iloc[[-1]]
latest_scaled = scaler.transform(latest_features)
tomorrow_lr = float(lr.predict(latest_scaled)[0])
tomorrow_rf = float(rf.predict(latest_features)[0])
latest_close = float(df_model['Close'].iloc[-1])
latest_date = df_model['Date'].iloc[-1]

print(f"\n{'='*60}")
print(f"Latest Close ({latest_date.date()}): ${latest_close:.2f}")
print(f"Predicted Next-Day Close (Linear Regression): ${tomorrow_lr:.2f}")
print(f"Predicted Next-Day Close (Random Forest):      ${tomorrow_rf:.2f}")
print(f"{'='*60}\n")

# ── 6. BUILD OUTPUT DATAFRAMES ──────────────────────────────────────────────

# 6a. Historical data with indicators (for charts in Power BI)
hist_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
              'MA7', 'MA21', 'MA50', 'Daily Return', 'RSI', 'MACD', 'Signal Line', 'Signal']
df_history = df_full[hist_cols].copy()

# 6b. Predictions vs Actual (test set)
df_predictions = pd.DataFrame({
    'Date': dates_test,
    'Actual Close': y_test.reset_index(drop=True),
    'Predicted (Linear Regression)': lr_pred,
    'Predicted (Random Forest)': rf_pred,
})
df_predictions['LR Error'] = df_predictions['Actual Close'] - df_predictions['Predicted (Linear Regression)']
df_predictions['RF Error'] = df_predictions['Actual Close'] - df_predictions['Predicted (Random Forest)']

# 6c. Model performance summary
df_metrics = pd.DataFrame([lr_metrics, rf_metrics])

# 6d. Next day forecast card
df_forecast = pd.DataFrame([{
    'Ticker': TICKER,
    'Last Updated': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'Latest Date': latest_date.strftime("%Y-%m-%d"),
    'Latest Close': round(latest_close, 2),
    'Predicted Next Close (LR)': round(tomorrow_lr, 2),
    'Predicted Next Close (RF)': round(tomorrow_rf, 2),
    'LR Expected Change %': round((tomorrow_lr - latest_close) / latest_close * 100, 2),
    'RF Expected Change %': round((tomorrow_rf - latest_close) / latest_close * 100, 2),
    'Current Signal': df_model['Signal'].iloc[-1],
    'Latest RSI': round(float(df_model['RSI'].iloc[-1]), 2),
}])

# 6e. Feature importance (Random Forest)
df_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

# ── 7. EXPORT TO EXCEL ───────────────────────────────────────────────────────
output_file = f"{TICKER}_AI_Dashboard_Data.xlsx"
print(f"Exporting to {output_file}...")

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    df_forecast.to_excel(writer, sheet_name='Forecast', index=False)
    df_history.to_excel(writer, sheet_name='Historical Data', index=False)
    df_predictions.to_excel(writer, sheet_name='Predictions', index=False)
    df_metrics.to_excel(writer, sheet_name='Model Metrics', index=False)
    df_importance.to_excel(writer, sheet_name='Feature Importance', index=False)

print(f"\nDone! Open '{output_file}' or connect Power BI to it.")
print("Re-run this script anytime to refresh with the latest live data.")
