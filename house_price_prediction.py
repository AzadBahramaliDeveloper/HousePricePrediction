import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Display basic info
print(train_df.info())
print(train_df.describe())

# Check for missing values
print(train_df.isnull().sum())

# Handle categorical columns
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

# Scale the selected numerical columns
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