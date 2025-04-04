import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

# Load the data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Display basic info for the training set
print(train_df.info())
print(train_df.describe())
train_df.dropna(inplace=True)

# Check for missing values
print(train_df.isnull().sum())

# Separate categorical and numerical columns
numerical_columns = train_df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = train_df.select_dtypes(include=['object']).columns

# Fill missing values for numerical columns with the mean
train_df[numerical_columns] = train_df[numerical_columns].fillna(train_df[numerical_columns].mean())

# Fill missing values for categorical columns with the mode (most frequent value)
train_df[categorical_columns] = train_df[categorical_columns].fillna(train_df[categorical_columns].mode().iloc[0])

# Handle categorical columns (dummy encoding)
train_df = pd.get_dummies(train_df, columns=['POSTED_BY', 'BHK_OR_RK'], drop_first=True)

# Display the updated dataset
print(train_df.head())

# Drop the 'ADDRESS' column
train_df.drop('ADDRESS', axis=1, inplace=True)

print(f"Shape after processing: {train_df.shape}")

# Initialize the scaler
scaler = StandardScaler()

# Select numerical columns to scale
numerical_columns = ['SQUARE_FT', 'LONGITUDE', 'LATITUDE']

# Scale the selected numerical columns in the training set
train_df[numerical_columns] = scaler.fit_transform(train_df[numerical_columns])

# Display the scaled dataset
print(train_df.head())

# Check if the 'TARGET(PRICE_IN_LACS)' column exists
if 'TARGET(PRICE_IN_LACS)' in train_df.columns:
    y = train_df['TARGET(PRICE_IN_LACS)']
    X = train_df.drop('TARGET(PRICE_IN_LACS)', axis=1)

    # Split the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Display the shapes of the resulting sets
    print(f"Training set (X_train) shape: {X_train.shape}")
    print(f"Test set (X_test) shape: {X_test.shape}")
else:
    print("TARGET(PRICE_IN_LACS) column is missing. Please check your dataset.")

# Apply the same transformations to the test set
test_df = pd.get_dummies(test_df, columns=['POSTED_BY', 'BHK_OR_RK'], drop_first=True)
test_df.drop('ADDRESS', axis=1, inplace=True)

# Apply the same scaling transformation to the test set (use the same scaler from the training set)
test_df[numerical_columns] = scaler.transform(test_df[numerical_columns])

# Initialize a RandomForestRegressor for reproducibility
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predictions on the test set (X_test)
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Model performance evaluation
if r2 < 0.7:
    print("Model performance is not satisfactory. Consider improving the feature engineering, adding more data, or trying other models.")
else:
    print("The model is performing well with R² of", r2)

# Hyperparameters to tune (RandomizedSearchCV)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

# Randomized Search CV
search = RandomizedSearchCV(RandomForestRegressor(), param_grid, cv=5, random_state=42)
search.fit(X_train, y_train)

print("Best parameters found: ", search.best_params_)

# Save model
joblib.dump(model, 'house_price_model.pkl')

# Load model
model = joblib.load('house_price_model.pkl')

# Evaluate on the test data and display the results
y_test_pred = model.predict(test_df)

# Display the results
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# Feature importances
feature_importances = model.feature_importances_
sorted_idx = feature_importances.argsort()
for index in sorted_idx:
    print(f"{X.columns[index]}: {feature_importances[index]}")

# Define the model
model = LogisticRegression()

# Define the hyperparameters to tune
param_grid = {
    'C': [0.1, 1, 10],  # Regularization strength
    'solver': ['liblinear', 'saga'],  # Solver options
    'max_iter': [100, 200, 500]  # Number of iterations
}

# Set up the grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found: ", grid_search.best_params_)

# Train the model with the best parameters
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')


# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_train)

# Standardize the new features
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)

# Split the new data into train and test
X_train_poly, X_test_poly, y_train, y_test = train_test_split(X_poly_scaled, y, test_size=0.2, random_state=42)

# Train a new model with polynomial features
model_poly = LogisticRegression(max_iter=200)
model_poly.fit(X_train_poly, y_train)

# Predict on the test set
y_pred_poly = model_poly.predict(X_test_poly)

# Evaluate the model
print(f'Accuracy with Polynomial Features: {accuracy_score(y_test, y_pred_poly)}')
print(f'Classification Report with Polynomial Features:\n{classification_report(y_test, y_pred_poly)}')
