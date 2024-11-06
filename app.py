# Importing necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data
# X represents the input (independent) variable
# y represents the output (dependent) variable
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # reshaping for a single feature
y = np.array([1, 3, 3, 2, 5])

# Create the model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plotting the results
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Linear Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Displaying the coefficients
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
