
# Healthcare Cost Prediction

This project demonstrates a predictive model for estimating total healthcare costs per patient based on demographic and medical history data. It simulates a real-world business case, especially relevant to actuarial analysts or professionals transitioning into data science.

## Dataset

A synthetic dataset (`healthcare_cost_data.csv`) was created with the following columns:
- `Patient_ID`
- `Age`
- `Gender`
- `Number_of_Visits`
- `Chronic_Conditions`
- `Total_Cost` (target variable)

## Goals

- Explore healthcare cost patterns
- Build machine learning models to predict patient-level costs
- Interpret model performance and insights

## Tools & Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- Seaborn, Matplotlib

## Models Used

- **Linear Regression**
- **Random Forest Regressor**

## Evaluation Metrics

- **Mean Absolute Error (MAE)**
- **R² Score (Explained Variance)**

## Visuals

The project includes:
- Distribution of healthcare costs
- Correlation heatmap
- Cost variation by gender

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
- Model performance output
- Prediction CSV file
- Visualizations

## Author

Matina Kpassira  
Analytics Enthusiast | Transitioning to Data Science
