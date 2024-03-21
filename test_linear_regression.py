from sklearn.linear_model import LinearRegression
import pandas as pd

X_train = pd.read_csv('insurance_features.txt', sep=' ', skiprows=1, header=None)
y_train = pd.read_csv('insurance_target.txt', sep=' ', header=None)

lr = LinearRegression()
lr.fit(X_train, y_train)

print("Coefficients & intercept: ")
print(lr.coef_)
print(lr.intercept_)

from sklearn.metrics import mean_absolute_error
print("Train Mean Absolute Error: ")
print(mean_absolute_error(lr.predict(X_train), y_train))

X_test = pd.read_csv('insurance_features_test.txt', sep=' ', skiprows=1, header=None)
y_test = pd.read_csv('insurance_target_test.txt', sep=' ', header=None)

print("Test Mean Absolute Error: ")
print(mean_absolute_error(lr.predict(X_test), y_test))