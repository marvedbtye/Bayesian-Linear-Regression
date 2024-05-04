# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:24:30 2024

@author: Carlos Mondejar
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
true_slope = 2.5
true_intercept = 5.0
y = true_slope * X + true_intercept + np.random.randn(100, 1) * 2.0

# Define Bayesian regression model
class BayesianLinearRegression:
    def __init__(self):
        self.w_mean = None
        self.w_cov = None

    def fit(self, X, y, alpha=1.0, beta=1.0):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        self.w_cov = np.linalg.inv(alpha * np.eye(X_b.shape[1]) + beta * X_b.T.dot(X_b))
        self.w_mean = beta * self.w_cov.dot(X_b.T.dot(y)).flatten()

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        return X_b.dot(self.w_mean)

# Fit the Bayesian model
blr = BayesianLinearRegression()
blr.fit(X, y)

# Plot the data and regression line
plt.scatter(X, y, label='Data Points')
plt.plot(X, blr.predict(X), color='red', label='Bayesian Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Bayesian Linear Regression')
plt.legend()
plt.grid(True)
plt.show()

# Display the parameters of the Bayesian model
print("Bayesian Linear Regression Parameters:")
print("Intercept:", blr.w_mean[0])
print("Slope:", blr.w_mean[1])

