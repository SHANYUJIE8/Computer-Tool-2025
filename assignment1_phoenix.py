#Group Name: Phoenix
#Member1:
#Name: Shan YUJIE Number: 2180790 Email:shan.2180790@studenti.uniroma1.it
#Member2:
#Name: Sun YUNYANG Number: 2190190 Email:sun.2190190@studenti.uniroma1.it
#Member3:
#Name: KONG YAEWEN Number:2190124 Email:kong.2190124@studenti.uniroma1.it
#Member4:
#Name: RU MENGYU Number:2179930	Email:ru.2179930@studenti.uniroma1.it

import pandas as pd
from numpy.linalg import solve
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/current.csv?sc_lang=en&hash=80445D12401C59CF716410F3F7863B64')

# Ensure sasdate is in datetime format
df['sasdate'] = pd.to_datetime(df['sasdate'], format='%m/%d/%Y', errors='coerce')

# Clean the DataFrame by removing the row with transformation codes
df_cleaned = df.drop(index=0)
df_cleaned.reset_index(drop=True, inplace=True)

## df_cleaned contains the data cleaned
df_cleaned

# Extract transformation codes
transformation_codes = df.iloc[0, 1:].to_frame().reset_index()
transformation_codes.columns = ['Series', 'Transformation_Code']

# Function to apply transformations based on the transformation code
def apply_transformation(series, code):
    series = series.astype(float)
    if code == 1:
        return series
    elif code == 2:
        return series.diff()
    elif code == 3:
        return series.diff().diff()
    elif code == 4:
        return np.log(series)
    elif code == 5:
        return np.log(series).diff()
    elif code == 6:
        return np.log(series).diff().diff()
    elif code == 7:
        return series.pct_change()
    else:
        raise ValueError("Invalid transformation code")

# Applying the transformations to each column in df_cleaned based on transformation_codes
for series_name, code in transformation_codes.values:
    df_cleaned[series_name] = apply_transformation(df_cleaned[series_name].astype(float), float(code))

df_cleaned.head()

############################################################################################################
## Create y and X for estimation of parameters
############################################################################################################

Yraw = df_cleaned['INDPRO']
Xraw = df_cleaned[['CPIAUCSL', 'TB3MS']]

## Number of lags and leads
num_lags  = 4  ## this is p
num_leads = 1  ## this is h

X = pd.DataFrame()
## Add the lagged values of Y
col = 'INDPRO'
for lag in range(0,num_lags+1):
        X[f'{col}_lag{lag}'] = Yraw.shift(lag)
## Add the lagged values of X
for col in Xraw.columns:
    for lag in range(0,num_lags+1):
        X[f'{col}_lag{lag}'] = Xraw[col].shift(lag)
## Add a column on ones (for the intercept)
X.insert(0, 'Ones', np.ones(len(X)))

## Remove NaN values to avoid errors
X.dropna(inplace=True)
y = Yraw.shift(-num_leads).loc[X.index]  # Ensure y only contains valid indices

############################################################################################################
## Estimation and forecast
############################################################################################################

## Save last row of X (converted to numpy)
X_T = X.iloc[-1:].values

## Subset getting only rows of X and y from p+1 to h-1
## and convert to numpy array
y = y.iloc[num_lags:-num_leads].values
X = X.iloc[num_lags:-num_leads].values
X_T

## Import the solve function from numpy.linalg
from numpy.linalg import solve

# Check for singular matrix issue and use normal equation method
try:
    beta_ols = solve(X.T @ X, X.T @ y)
except np.linalg.LinAlgError:
    beta_ols = np.linalg.pinv(X.T @ X) @ X.T @ y  # Use pseudo-inverse if singular

## Produce the One step ahead forecast
## % change month-to-month of INDPRO
forecast = X_T @ beta_ols * 100
forecast

############################################################################################################
## Improved function to handle missing future dates
############################################################################################################
def calculate_forecast(df_cleaned, p=4, H=[1,4,8], end_date='12/1/1999', target='INDPRO', xvars=['CPIAUCSL', 'TB3MS']):
    ## Ensure sasdate is in datetime format
    df_cleaned['sasdate'] = pd.to_datetime(df_cleaned['sasdate'], errors='coerce')
    
    ## Subset df_cleaned to use only data up to end_date
    end_date = pd.Timestamp(end_date)
    rt_df = df_cleaned[df_cleaned['sasdate'] <= end_date]
    
    ## Get the actual values of target at different steps ahead
    Y_actual = []
    for h in H:
        os = pd.Timestamp(end_date) + pd.DateOffset(months=h)
        closest_date = df_cleaned[df_cleaned['sasdate'] >= os]['sasdate'].min()
        actual_value = df_cleaned[df_cleaned['sasdate'] == closest_date][target] * 100
        Y_actual.append(actual_value.values[0] if not actual_value.empty else np.nan)
    
    Yraw = rt_df[target]
    Xraw = rt_df[xvars]
    
    X = pd.DataFrame()
    for lag in range(0, p):
        X[f'{target}_lag{lag}'] = Yraw.shift(lag)
    for col in Xraw.columns:
        for lag in range(0, p):
            X[f'{col}_lag{lag}'] = Xraw[col].shift(lag)
    X.insert(0, 'Ones', np.ones(len(X)))
    X.dropna(inplace=True)
    
    X_T = X.iloc[-1:].values
    
    Yhat = []
    for h in H:
        y_h = Yraw.shift(-h)
        valid_rows = (~X.isna().any(axis=1)) & (~y_h.isna())
        X_valid = X.loc[valid_rows].values
        y_valid = y_h.loc[valid_rows].values
        if len(y_valid) > 0:
            try:
                beta_ols = solve(X_valid.T @ X_valid, X_valid.T @ y_valid)
            except np.linalg.LinAlgError:
                beta_ols = np.linalg.pinv(X_valid.T @ X_valid) @ X_valid.T @ y_valid
            forecast_value = X_T @ beta_ols * 100
            Yhat.append(forecast_value[0])
        else:
            Yhat.append(np.nan)
    
    return np.array(Y_actual), np.array(Yhat)

## Run forecast calculations
t0 = pd.Timestamp('12/1/1999')
e = []
T = []
Y_actual_all = []
Yhat_all = []
for j in range(10):
    t0 = t0 + pd.DateOffset(months=1)
    Y_actual, Yhat = calculate_forecast(df_cleaned, p=4, H=[1,4,8], end_date=t0)
    e.append((Y_actual - Yhat).flatten())
    T.append(t0)
    Y_actual_all.append(Y_actual)
    Yhat_all.append(Yhat)

## Create DataFrame and compute RMSFE
edf = pd.DataFrame(e)
rmsf = np.sqrt(edf.apply(np.square).mean())

print(forecast)
print(rmsf)

## Plot actual vs forecast
plt.figure(figsize=(10, 6))
for i, h in enumerate([1, 4, 8]):
    plt.plot(T, [y[i] for y in Y_actual_all], label=f'Actual (H={h})')
    plt.plot(T, [y[i] for y in Yhat_all], linestyle='--', label=f'Forecast (H={h})')

plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Actual vs Forecast')
plt.legend()
plt.show()