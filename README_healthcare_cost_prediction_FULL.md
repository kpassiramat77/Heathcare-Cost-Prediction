
# Healthcare Cost Prediction

This project demonstrates a predictive model for estimating total healthcare costs per patient based on demographic and medical history data. It simulates a real-world business case, especially relevant to actuarial analysts or professionals transitioning into data science.

---

## Dataset

A synthetic dataset (`healthcare_cost_data.csv`) was created with the following columns:
- `Patient_ID`
- `Age`
- `Gender`
- `Number_of_Visits`
- `Chronic_Conditions`
- `Total_Cost` (target variable)

---

## Goals

- Explore patterns in healthcare cost using statistical and visual analysis
- Build and compare machine learning models to predict patient-level healthcare costs
- Provide insights on feature importance and business impact

---

## Tools & Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib

---

## Data Analysis

Performed exploratory data analysis (EDA) to understand relationships between patient characteristics and cost:
- **Distribution Plot** of total healthcare cost
- **Correlation Heatmap** between cost drivers (age, visits, chronic conditions)
- **Boxplot** comparing cost distribution by gender

These insights informed feature selection and model expectations.

---

## Predictive Modeling

Two regression models were used:
- **Linear Regression**
- **Random Forest Regressor**

**Features Used**:
- Age
- Gender (encoded)
- Number of Visits
- Chronic Conditions

**Evaluation Metrics**:
- **Mean Absolute Error (MAE)**
- **R² Score (Explained Variance)**

Example Output:
- Linear Regression: MAE ~ $250, R² ~ 0.12
- Random Forest: MAE ~ $258, R² ~ 0.02

Visualizations and predictions were saved to CSV for comparison and further review.

---

## How to Run

1. Clone the repository
2. Install required libraries:
    ```
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
3. Run the Python script:
    ```
    python healthcare_cost_prediction_project.py
    ```

This will generate:
- Model evaluation results in console
- Visualizations (`.png`)
- Predictions CSV output

---

## Author

**Matina Kpassira**  
Analytics Enthusiast | Transitioning to Data Science  
Kent, WA | kpassiramat@gmail.com
