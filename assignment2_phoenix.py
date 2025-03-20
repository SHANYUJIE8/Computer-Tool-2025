#Group Name: Phoenix
#Member1:
#Name: Shan YUJIE Number: 2180790 Email:shan.2180790@studenti.uniroma1.it
#Member2:
#Name: Sun YUNYANG Number: 2190190 Email:sun.2190190@studenti.uniroma1.it
#Member3:
#Name: KONG YAEWEN Number:2190124 Email:kong.2190124@studenti.uniroma1.it
#Member4:
#Name: RU MENGYU Number:2179930	Email:ru.2179930@studenti.uniroma1.it
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

# Load the dataset
file_path = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv"
df = pd.read_csv(file_path)

# Remove the first row (transformation codes)
df = df.iloc[1:].reset_index(drop=True)

# Convert 'sasdate' to datetime format
df['sasdate'] = pd.to_datetime(df['sasdate'])

# Convert 'INDPRO' to numeric
df['INDPRO'] = pd.to_numeric(df['INDPRO'], errors='coerce')

# Drop NaN values
df = df[['sasdate', 'INDPRO']].dropna()

def ar2_loglikelihood(params, y, conditional=True):
    """Compute the log-likelihood for an AR(2) model."""
    c, phi1, phi2, sigma2 = params
    
    # Ensure stationarity constraints
    if not (-1 < phi2 < 1 and -1 < phi1 + phi2 < 1 and -1 < phi1 - phi2 < 1):
        return np.inf
    
    T = len(y)
    if T < 3:
        raise ValueError("Time series must have at least 3 observations for AR(2)")
    
    residuals = np.zeros(T)
    if conditional:
        residuals[2:] = y[2:] - (c + phi1 * y[1:-1] + phi2 * y[:-2])
    else:
        # Assume Y1 and Y2 follow a stationary distribution
        residuals[:] = y - (c + phi1 * np.roll(y, 1) + phi2 * np.roll(y, 2))
    
    loglik = -T / 2 * np.log(2 * np.pi * sigma2) - np.sum(residuals**2) / (2 * sigma2)
    return -loglik

def fit_ar2_mle(y, conditional=True):
    """Fit an AR(2) model using Maximum Likelihood Estimation (MLE)."""
    
    # Use Yule-Walker to initialize phi1 and phi2
    ar_model = AutoReg(y, lags=2).fit()
    initial_params = np.array([ar_model.params[0], ar_model.params[1], ar_model.params[2], np.var(y)])
    
    # Bounds for parameters
    bounds = [(-np.inf, np.inf), (-0.99, 0.99), (-0.99, 0.99), (1e-6, np.inf)]
    
    # Optimize log-likelihood function
    result = minimize(ar2_loglikelihood, initial_params, args=(y, conditional),
                      bounds=bounds, method='L-BFGS-B',
                      options={'maxiter': 1000})  # Increase max iterations
    
    if not result.success:
        print(f"Warning: Optimization did not converge. {result.message}")
    
    return result.x  # Return estimated parameters

def predict_ar2(y, params, h=8):
    """Predict future values using an estimated AR(2) model."""
    c, phi1, phi2, sigma2 = params
    y_pred = list(y[-2:])
    
    for _ in range(h):
        next_value = c + phi1 * y_pred[-1] + phi2 * y_pred[-2]
        y_pred.append(next_value)
    
    return np.array(y_pred[2:])

# Prepare data
data = df['INDPRO'].values
params = fit_ar2_mle(data, conditional=True)
print("Estimated AR(2) parameters:", params)

# Predict next 8 months
y_pred = predict_ar2(data, params, h=8)
print("Predicted future values:", y_pred)

# Compute confidence intervals
pred_std = np.sqrt(params[3])
upper_bound = y_pred + 1.96 * pred_std
lower_bound = y_pred - 1.96 * pred_std

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(df['sasdate'], df['INDPRO'], label="Observed Data")
pred_dates = pd.date_range(start=df['sasdate'].iloc[-1], periods=9, freq='M')[1:]
plt.plot(pred_dates, y_pred, 'r--', label="Predicted")
plt.fill_between(pred_dates, lower_bound, upper_bound, color='r', alpha=0.2, label="95% CI")
plt.xlabel("Time")
plt.ylabel("INDPRO")
plt.legend()
plt.title("AR(2) Model Prediction for INDPRO")
plt.grid(True)
plt.tight_layout()
plt.show()