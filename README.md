# 🔥 Calories Burned Predictor

Machine learning model that predicts the number of calories burned during a gym session based on member demographics, biometrics, and workout characteristics.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange)
![R²](https://img.shields.io/badge/R²-0.969-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This project builds a regression model on the **Gym Members Exercise Dataset** to estimate `Calories_Burned` from a combination of personal attributes and session metrics.  
The final model is a tuned **RandomForestRegressor** that achieves an **R² ≈ 0.969** on the held-out test set.

## Dataset

**Source:** [Gym Members Exercise Dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset) (Kaggle)  
973 records · 15 columns

### Features

| Feature                        | Type        | Description                          |
|--------------------------------|-------------|--------------------------------------|
| `Age`                          | Numeric     | Age of the member                    |
| `Gender`                       | Categorical | Male / Female                        |
| `Weight (kg)`                  | Numeric     | Body weight                          |
| `Height (m)`                   | Numeric     | Height                               |
| `Max_BPM`                      | Numeric     | Maximum heart rate during session    |
| `Avg_BPM`                      | Numeric     | Average heart rate                   |
| `Resting_BPM`                  | Numeric     | Resting heart rate                   |
| `Session_Duration (hours)`     | Numeric     | Length of the workout                |
| `Workout_Type`                 | Categorical | Yoga, HIIT, Cardio, Strength         |
| `Fat_Percentage`               | Numeric     | Body fat percentage                  |
| `Water_Intake (liters)`        | Numeric     | Water consumed                       |
| `Workout_Frequency (days/week)`| Numeric     | Training frequency                   |
| `Experience_Level`             | Numeric     | 1–3 experience scale                 |
| `BMI`                          | Numeric     | Body Mass Index                      |

**Target:** `Calories_Burned`

## Methodology

### 1. Preprocessing
- Custom **One-Hot Encoder** for `Gender` and `Workout_Type`
- **StandardScaler** applied to all features after encoding
- No missing values in the dataset

### 2. Modeling
- Base estimator: `RandomForestRegressor`
- Hyperparameter tuning with both `GridSearchCV` and `RandomizedSearchCV`
- Best model selected from the randomized search

### Best Hyperparameters
```python
RandomForestRegressor(
    max_depth=10,
    max_features=None,
    min_samples_leaf=5,
    n_estimators=213,
    n_jobs=-1
)
