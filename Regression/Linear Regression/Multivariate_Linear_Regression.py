"""
Script for Predicting Insurance Charges Based on Various Factors
This script imports the required libraries, loads the dataset, and processes it through steps like handling null values,
scaling features, encoding categorical variables, splitting the dataset, training a regression model, and evaluating model performance.
"""

import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Step 1: Load the insurance dataset
dataset = pd.read_csv("../insurance_dataset.csv")

# Step 2: Inspect dataset details (structure and data types)
dataset.info()

# Step 3: Get descriptive statistics for numerical features
dataset.describe()

# Step 4: Check for null values in the dataset
dataset.isnull().sum()

# Step 5: Fill missing values in specific columns with default values
dataset["medical_history"] = dataset["medical_history"].fillna("No Disease")
dataset["family_medical_history"] = dataset["family_medical_history"].fillna(
    "No History"
)


# Step 6: Define function to remove outliers based on Z-score
def remove_outliers(df, column):
    """
    Removes outliers from a specified column in a DataFrame based on Z-scores.
    Args:
    - df (DataFrame): Input DataFrame
    - column (str): Column from which to remove outliers
    Returns:
    - DataFrame without outliers in the specified column
    """
    z_scores = stats.zscore(df[column])
    return df[(abs(z_scores) <= 3.4)]


# Columns requiring outlier removal
Outlier_remove = [
    "children",
    "smoker",
    "region",
    "medical_history",
    "family_medical_history",
    "exercise_frequency",
    "coverage_level",
]

# Remove outliers for each group defined in `Outlier_remove`
for o in Outlier_remove:
    dataset = (
        dataset.groupby(o)
        .apply(lambda x: remove_outliers(x, "charges"))
        .reset_index(drop=True)
    )

# Step 7: Scale numerical features that require normalization
needs_to_be_scaled = ["age", "bmi", "children"]
for feature in needs_to_be_scaled:
    scaler = StandardScaler()
    dataset[feature] = scaler.fit_transform(dataset[[feature]])

# Step 8: Apply binary class encoding to features with two classes
binary_class_feature = ["smoker", "gender"]
for feature in binary_class_feature:
    encoder = LabelEncoder()
    dataset[feature] = encoder.fit_transform(dataset[feature])

# Step 9: Apply one-hot encoding to features with multiple classes
multi_class_feature = [
    "region",
    "medical_history",
    "family_medical_history",
    "exercise_frequency",
    "occupation",
    "coverage_level",
]

for feature in multi_class_feature:
    encoder = OneHotEncoder(sparse_output=False)
    encoded_feature = encoder.fit_transform(dataset[[feature]])
    dataset = dataset.drop(columns=[feature])
    encoded_df = pd.DataFrame(
        encoded_feature, columns=encoder.get_feature_names_out([feature])
    )
    dataset = pd.concat([dataset, encoded_df], axis=1)

# Step 10: Define target variable and features
X = dataset.drop("charges", axis="columns")
Y = pd.DataFrame(dataset["charges"])

# Step 11: Split data into training and testing sets (75% train, 25% test)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=103
)

# Step 12: Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Step 13: Make predictions on both train and test datasets
y_predicted_train = model.predict(X_train)
y_predicted_test = model.predict(X_test)

# Step 14: Evaluate model performance on the training dataset
train_error_r2 = r2_score(Y_train, y_predicted_train)
train_error_mse = mean_squared_error(Y_train, y_predicted_train)
print(f"Train Error:\nMean Square Error: {train_error_mse}\nR2 Score: {train_error_r2}")

# Step 15: Evaluate model performance on the test dataset
test_error_r2 = r2_score(Y_test, y_predicted_test)
test_error_mse = mean_squared_error(Y_test, y_predicted_test)
print(f"Test Error:\nMean Square Error: {test_error_mse}\nR2 Score: {test_error_r2}")

# Step 16: Calculate the overall model score on the test dataset
score = model.score(X_test, Y_test)
print("Model Score on Test Dataset:", score)
