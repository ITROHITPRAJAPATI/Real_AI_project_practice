# Employee Attrition Prediction

## Project Overview

This is a Machine Learning project that predicts whether an employee is likely to leave the company or stay.

## Objective

The main objective of this project is to predict employee attrition based on factors such as age, job satisfaction, monthly income, years at company, overtime, and job level.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Decision Tree Classifier
- Joblib

## Features Used

- Age
- Job Satisfaction
- Monthly Income
- Years At Company
- OverTime
- Job Level

## Machine Learning Workflow

1. Load the employee dataset
2. Convert categorical values into numerical values
3. Select features and target
4. Split data into training and testing sets
5. Train Decision Tree Classifier
6. Evaluate model accuracy
7. Save trained model
8. Load model for prediction
9. Predict employee attrition

## Prediction

The model predicts:

- 1 → Employee may leave the company
- 0 → Employee may stay in the company

## Project Structure

employee-attrition-prediction/
│
├── data/
│   └── employee_attrition.csv
│
├── models/
│   └── employee_attrition_model.pkl
│
├── train.py
├── predict.py
└── README.md