import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

# Set random seed for reproducibility
np.random.seed(0)

# Generate data
x = np.linspace(0, 10, 100).reshape(-1, 1)
epsilon = np.random.normal(0, np.sqrt(4), x.shape)  # Gaussian noise with variance 4
y = 2 * x + epsilon

# Fit a linear regression model
linear_model = LinearRegression()
linear_model.fit(x, y)
y_pred_linear = linear_model.predict(x)

# Fit a 9th-degree polynomial regression model
polynomial_model = make_pipeline(PolynomialFeatures(9), LinearRegression())
polynomial_model.fit(x, y)
y_pred_poly = polynomial_model.predict(x)

# Calculate training errors (Mean Squared Error)
mse_linear = mean_squared_error(y, y_pred_linear)
mse_poly = mean_squared_error(y, y_pred_poly)

# Plot the results
plt.figure(figsize=(14, 6))

# Original data
plt.scatter(x, y, color='gray', alpha=0.5, label='Data')

# Linear regression fit
plt.plot(x, y_pred_linear, color='blue', label=f'Linear Fit (MSE: {mse_linear:.2f})')

# Polynomial regression fit
plt.plot(x, y_pred_poly, color='red', linestyle='--', label=f'9th-Degree Polynomial Fit (MSE: {mse_poly:.2f})')

# Labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression vs 9th-Degree Polynomial Regression')
plt.legend()
plt.show()

print("Linear Regression MSE:", mse_linear)
print("9th-Degree Polynomial Regression MSE:", mse_poly)
