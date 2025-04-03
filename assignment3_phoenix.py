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

# Summary:
# In this assignment, we used Monte Carlo simulations and the Moving Block Bootstrap method to assess the accuracy 
# of inference for  \beta_1  in a regression model with AR(1) errors. 
# When the sample size was  T = 100 , 
# the bootstrap confidence interval captured the true value in approximately 78% to 82% of the simulations, 
# which is below the nominal 95% level. 
# When the sample size increased to  T = 500 , the coverage rate improved to around 91% to 94%. 
# These results indicate that while the bootstrap method is useful under serial correlation, 
# it may underestimate uncertainty in small samples, leading to undercoverage of the true parameter.

import numpy as np

# Function to simulate an AR(1) process
def simulate_ar1(n, phi, sigma):
    errors = np.zeros(n)
    eta = np.random.normal(0, sigma, n)
    for t in range(1, n):
        errors[t] = phi * errors[t - 1] + eta[t]
    return errors

# Simulate a regression model with AR(1) errors
def simulate_regression_with_ar1_errors(n, beta0, beta1, phi_x, phi_u, sigma):
    x = simulate_ar1(n, phi_x, sigma)
    u = simulate_ar1(n, phi_u, sigma)
    y = beta0 + beta1 * x + u
    return x, y, u

# Parameter settings
T = 100              # Sample size
beta0 = 1            # Intercept
beta1 = 2            # Slope (target parameter)
phi_x = 0.7          # Autocorrelation for x
phi_u = 0.7          # Autocorrelation for u (errors)
sigma = 1            # Standard deviation of white noise

# Step 1: Simulate one dataset
x, y, u = simulate_regression_with_ar1_errors(T, beta0, beta1, phi_x, phi_u, sigma)

import matplotlib.pyplot as plt
# plt.plot(x, label="x (AR(1))")
# plt.plot(u, label="u (AR(1))")
# plt.plot(y, label="y")
# plt.legend()
# plt.title("Simulated AR(1) Data")
# plt.show()

import statsmodels.api as sm

# Moving Block Bootstrap function
def moving_block_bootstrap(x, y, block_length, num_bootstrap):
    T = len(y)
    num_blocks = T // block_length + (1 if T % block_length else 0)
    bootstrap_estimates = np.zeros(num_bootstrap)

    for i in range(num_bootstrap):
        # Draw starting points for each bootstrap block
        block_starts = np.random.choice(np.arange(num_blocks) * block_length, size=num_blocks, replace=True)
        sample_indices = np.hstack([
            np.arange(start, min(start + block_length, T)) for start in block_starts
        ])[:T]  # Ensure bootstrap sample length equals original sample

        x_b = x[sample_indices]
        y_b = y[sample_indices]

        X_b = sm.add_constant(x_b)
        model_b = sm.OLS(y_b, X_b).fit()
        bootstrap_estimates[i] = model_b.params[1]  # Store beta_1

    return bootstrap_estimates

# Settings for bootstrap
block_length = 12
num_bootstrap = 500

# Run moving block bootstrap
bootstrap_beta1 = moving_block_bootstrap(x, y, block_length, num_bootstrap)

# Compute standard error (standard deviation of estimates)
bootstrap_se = np.std(bootstrap_beta1)
print("Bootstrap standard error (SE) for beta_1 =", bootstrap_se)

# Fit OLS once on original data to get beta_1 estimate
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
beta1_hat = model.params[1]

# Construct 95% bootstrap confidence interval
lower_bound = beta1_hat - 1.96 * bootstrap_se
upper_bound = beta1_hat + 1.96 * bootstrap_se

print("Bootstrap 95% CI for beta_1:", (lower_bound, upper_bound))

# Check whether true beta_1 = 2 is within the CI
if lower_bound <= 2 <= upper_bound:
    print("Bootstrap CI contains the true value beta_1 = 2")
else:
    print("Bootstrap CI does NOT contain beta_1 = 2")

# ==== Step 4: Monte Carlo Simulation ====
num_simulations100 = 100  # Number of repeated simulations
contain_true_value = []

for i in range(num_simulations100):
    # Generate new (x, y) each time
    x, y, u = simulate_regression_with_ar1_errors(T, beta0, beta1, phi_x, phi_u, sigma)

    # Fit OLS and get beta_1 estimate
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    beta1_hat = model.params[1]

    # Run bootstrap
    bootstrap_beta1 = moving_block_bootstrap(x, y, block_length, num_bootstrap)
    bootstrap_se = np.std(bootstrap_beta1)

    # Construct 95% bootstrap confidence interval
    lower = beta1_hat - 1.96 * bootstrap_se
    upper = beta1_hat + 1.96 * bootstrap_se

    # Check whether CI includes the true value beta_1 = 2
    if lower <= beta1 <= upper:
        contain_true_value.append(1)
    else:
        contain_true_value.append(0)

# Print final empirical coverage rate
coverage_rate = np.mean(contain_true_value)
print(f"\n🎯 Bootstrap CI coverage rate over {num_simulations100} simulations: {coverage_rate:.3f}")


T = 500  # Run Monte Carlo for T = 500
num_simulations500 = 500  # Number of repeated simulations
contain_true_value = []

for i in range(num_simulations500):
    # Generate new (x, y) each time
    x, y, u = simulate_regression_with_ar1_errors(T, beta0, beta1, phi_x, phi_u, sigma)

    # Fit OLS and get beta_1 estimate
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    beta1_hat = model.params[1]

    # Run bootstrap
    bootstrap_beta1 = moving_block_bootstrap(x, y, block_length, num_bootstrap)
    bootstrap_se = np.std(bootstrap_beta1)

    # Construct 95% bootstrap confidence interval
    lower = beta1_hat - 1.96 * bootstrap_se
    upper = beta1_hat + 1.96 * bootstrap_se

    # Check whether CI includes the true value beta_1 = 2
    if lower <= beta1 <= upper:
        contain_true_value.append(1)
    else:
        contain_true_value.append(0)

# Print final empirical coverage rate
coverage_rate = np.mean(contain_true_value)
print(f"\n🎯 Bootstrap CI coverage rate over {num_simulations500} simulations: {coverage_rate:.3f}")