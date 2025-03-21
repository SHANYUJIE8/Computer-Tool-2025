#Group Name: Phoenix
#Member1:
#Name: Shan YUJIE Number: 2180790 Email:shan.2180790@studenti.uniroma1.it
#Member2:
#Name: Sun YUNYANG Number: 2190190 Email:sun.2190190@studenti.uniroma1.it
#Member3:
#Name: KONG YAEWEN Number:2190124 Email:kong.2190124@studenti.uniroma1.it
#Member4:
#Name: RU MENGYU Number:2179930	Email:ru.2179930@studenti.uniroma1.it
#Member5:
#Name:LIANG	HUAMING	Number:2180952	Email:liang.2180952@studenti.uniroma1.it

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

# Load and preprocess INDPRO data
file_path = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv"
df = pd.read_csv(file_path)
df = df.iloc[1:].reset_index(drop=True)
df['sasdate'] = pd.to_datetime(df['sasdate'])
df['INDPRO'] = pd.to_numeric(df['INDPRO'], errors='coerce')
df = df[['sasdate', 'INDPRO']].dropna()

# Log-difference transformation for stationarity
log_diff = np.diff(np.log(df['INDPRO'].values))

# Define AR(2) log-likelihood
def ar2_loglikelihood(params, y):
    c, phi1, phi2, sigma2 = params
    if sigma2 <= 0 or not (-0.99 < phi1 < 0.99 and -0.99 < phi2 < 0.99):
        return np.inf
    T = len(y)
    residuals = y[2:] - (c + phi1 * y[1:-1] + phi2 * y[:-2])
    loglik = -0.5 * (T - 2) * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(residuals**2) / sigma2
    return -loglik

#  Fit AR(2) model using MLE
def fit_ar2_mle(y):
    model = AutoReg(y, lags=2, old_names=False).fit()
    initial = [model.params[0], model.params[1], model.params[2], np.var(y)]
    bounds = [(-np.inf, np.inf), (-0.99, 0.99), (-0.99, 0.99), (1e-6, np.inf)]
    result = minimize(ar2_loglikelihood, initial, args=(y,), bounds=bounds, method='L-BFGS-B')
    if not result.success:
        print("Warning: Optimization failed -", result.message)
    return result.x

# Forecast function
def predict_ar2(y, params, h=8):
    c, phi1, phi2, _ = params
    history = list(y[-2:])
    preds = []
    for _ in range(h):
        pred = c + phi1 * history[-1] + phi2 * history[-2]
        preds.append(pred)
        history.append(pred)
    return np.array(preds)

# Estimate model and forecast
params = fit_ar2_mle(log_diff)
print("Estimated AR(2) parameters:", params)

y_pred_diff = predict_ar2(log_diff, params, h=8)

#  Convert predictions back to INDPRO levels
last_value = df['INDPRO'].iloc[-1]
y_pred_levels = last_value * np.exp(np.cumsum(y_pred_diff))

#  Confidence intervals
sigma = np.sqrt(params[3])
upper_bound = y_pred_levels * np.exp(1.96 * sigma)
lower_bound = y_pred_levels * np.exp(-1.96 * sigma)

# Forecast dates
pred_dates = pd.date_range(start=df['sasdate'].iloc[-1], periods=9, freq='ME')[1:]

#  Print forecast results
print("\nPredicted INDPRO levels for the next 8 months:")
for i in range(8):
    print(f"Month {i+1} ({pred_dates[i].strftime('%Y-%m')}): {y_pred_levels[i]:.2f}")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df['sasdate'], df['INDPRO'], label='Observed INDPRO')
plt.axvline(df['sasdate'].iloc[-1], color='gray', linestyle='--', label='Forecast Start')
plt.plot(pred_dates, y_pred_levels, 'r--', label='Forecast')
plt.fill_between(pred_dates, lower_bound, upper_bound, color='red', alpha=0.2, label='95% CI')
plt.xlabel("Time")
plt.ylabel("INDPRO")
plt.title("AR(2) Forecast for INDPRO (Log-diff model)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
