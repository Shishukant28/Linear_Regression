import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

# Generate some sample data
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])  # Add noise

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear_pred = linear_model.predict(X)

# Polynomial Regression
poly_features = PolynomialFeatures(degree=4)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

# Support Vector Regression (SVR)
svr_model = SVR(kernel='rbf')
svr_model.fit(X, y)
y_svr_pred = svr_model.predict(X)

# Plotting
plt.figure(figsize=(12, 8))

# Scatter plot of original data
plt.scatter(X, y, color='black', label='Data', s=10)

# Plot Linear Regression
plt.plot(X, y_linear_pred, color='blue', label='Linear Regression')

# Plot Polynomial Regression
plt.plot(X, y_poly_pred, color='green', label='Polynomial Regression (degree=4)')

# Plot SVR
plt.scatter(X, y_svr_pred, color='red', label='Support Vector Regression', s=10)

# Adding titles and labels
plt.title('Regression Analysis')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid()

# Show the plot
plt.show()
