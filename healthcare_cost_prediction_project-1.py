
# Healthcare Cost Prediction Model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load data
df = pd.read_csv("healthcare_cost_data.csv")

# Encode categorical variables
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# EDA (Optional Visuals)
sns.histplot(df["Total_Cost"], bins=30, kde=True)
plt.title("Distribution of Total Healthcare Cost")
plt.xlabel("Total Cost")
plt.ylabel("Number of Patients")
plt.savefig("cost_distribution.png")
plt.clf()

sns.heatmap(df[["Age", "Number_of_Visits", "Chronic_Conditions", "Total_Cost"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("correlation_matrix.png")
plt.clf()

sns.boxplot(data=df, x="Gender", y="Total_Cost")
plt.title("Total Cost Distribution by Gender")
plt.savefig("gender_boxplot.png")
plt.clf()

# Define features and target
X = df[['Age', 'Gender', 'Number_of_Visits', 'Chronic_Conditions']]
y = df['Total_Cost']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Evaluation
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_r2 = r2_score(y_test, lr_preds)

rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)

# Print results
print("Linear Regression - MAE:", round(lr_mae, 2), "R²:", round(lr_r2, 2))
print("Random Forest - MAE:", round(rf_mae, 2), "R²:", round(rf_r2, 2))

# Save predictions for review
results = pd.DataFrame({
    "Actual": y_test,
    "LR_Predicted": lr_preds,
    "RF_Predicted": rf_preds
})
results.to_csv("predictions_output.csv", index=False)
