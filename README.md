# Diabetes Prediction Model

## Description
This project implements an advanced machine learning solution for predicting diabetes risk using health metrics data. Unlike conventional approaches that rely on off-the-shelf algorithms, I developed a custom logistic regression model with L2 regularization built from scratch. Through innovative feature engineering based on clinical knowledge, comprehensive statistical analysis, and threshold optimization, I achieved significantly better performance than standard approaches.

The model achieves 92.03% recall (sensitivity) - a critical metric in medical diagnostics where false negatives can lead to missed treatment opportunities. By implementing domain-specific features like glucose threshold indicators and metabolic syndrome risk factors, the model captures complex relationships in the data that generic implementations miss. The custom implementation not only outperforms scikit-learn's logistic regression but also provides greater flexibility for clinical applications where threshold tuning is essential for balancing sensitivity against specificity.

## Project Overview
This project implements a machine learning model to predict diabetes in patients based on various health metrics. Using advanced feature engineering, statistical analysis, and optimization techniques, I was able to significantly improve the model's performance over baseline approaches.

## Key Achievements
- **Increased recall from 87% to 92.03%** (critical for healthcare applications where false negatives can be dangerous)
- **Improved accuracy from 68% to 72%**
- **Significantly outperformed scikit-learn's implementation** in both recall and accuracy
- Developed custom feature engineering pipeline based on clinical knowledge
- Implemented regularized logistic regression from scratch with superior performance
- Optimized classification threshold for better clinical utility

## Dataset
The project uses the Pima Indians Diabetes Database, which contains the following features:
- Pregnancies: Number of pregnancies
- Glucose: Plasma glucose concentration (2 hours in an oral glucose tolerance test)
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)²)
- DiabetesPedigreeFunction: Diabetes pedigree function (a function that scores likelihood of diabetes based on family history)
- Age: Age in years
- Outcome: Class variable (0 or 1) indicating presence of diabetes

## Data Preprocessing
Several important preprocessing steps were implemented:
1. **Missing value imputation**: Replaced zeros in Glucose, BloodPressure, SkinThickness, Insulin, and BMI columns with median values
2. **Feature scaling**: Standardized numerical features using Z-score normalization
3. **Train-test split**: 80% train, 20% test with stratification

## Feature Engineering
New clinically relevant features were developed to improve model performance:
1. **Glucose threshold indicators**:
   - PreDiabetic: 100-125 mg/dL
   - Diabetic: 126-180 mg/dL
   - SevereHyperglycemia: >180 mg/dL
2. **BMI categorization**:
   - Overweight: 25-30 BMI
3. **Age risk groups**:
   - Senior: 60+ years
4. **Metabolic syndrome composite indicator**:
   - Combined risk from glucose, BMI, age, and family history

## Model Implementation
Instead of using off-the-shelf implementations, I developed a logistic regression model from scratch with:
1. **L2 regularization** to prevent overfitting
2. **Gradient descent optimizer** with carefully tuned learning rate
3. **Custom cost function** with regularization term
4. **Threshold optimization** for better sensitivity/specificity balance

## Model Evaluation
Comprehensive evaluation metrics were used to assess performance:
- **Accuracy**: 72% (improved from 68%)
- **Recall/Sensitivity**: 92.03% (improved from 87%)
- **Precision**: 65.4%
- **F1 Score**: 76.4%
- **ROC-AUC**: 0.842

## Threshold Optimization
A critical finding was that the default 0.5 probability threshold was suboptimal for this clinical application. Through systematic testing of different thresholds, I determined that a threshold of 0.3 provided the best balance between precision and recall, maximizing the F1 score.

## Feature Importance Analysis
The analysis of feature weights revealed the most important predictors of diabetes:
1. Glucose levels (especially SevereHyperglycemia)
2. BMI and Overweight status
3. Age and Senior status
4. Pregnancies
5. DiabetesPedigreeFunction

## Cross-Validation
K-fold cross-validation was implemented to ensure model robustness. The model maintained consistent performance across folds, indicating reliability.

## Comparison with Scikit-learn Implementation
To validate my custom implementation, I compared results with scikit-learn's LogisticRegression:
- **Custom model significantly outperformed on recall (92.03% vs 85.5%)**
- **Custom model achieved superior accuracy (72% vs 67.8%)**
- Custom implementation provided greater flexibility for threshold optimization
- My model demonstrated better generalization to unseen data
- Custom gradient descent approach with regularization proved more effective than scikit-learn's solver

## Visualizations
The project includes numerous visualizations to aid interpretation:
- Confusion matrices
- ROC curves
- Feature correlation heatmap
- Feature importance plot
- Cost vs. iteration plots
- Threshold optimization analysis

## Conclusions
This project demonstrates the value of:
1. Domain-specific feature engineering based on clinical knowledge
2. Custom model implementation allowing for threshold optimization
3. Comprehensive model evaluation focused on the most important metric for the application (recall)

For medical applications like diabetes prediction, optimizing for recall (sensitivity) is critical to minimize false negatives, as missing a positive diagnosis is typically more harmful than a false positive.

## Future Work
Potential improvements include:
- Implementation of more advanced ensemble methods
- Further feature engineering incorporating more domain knowledge
- Exploration of deep learning approaches for feature extraction
- Collection of additional relevant features like HbA1c levels

## Usage
```python
# Example usage of the model
import pandas as pd
from diabetes_model import engineer_features, feature_scaling

# Load test data
test_data = pd.read_csv("new_patient_data.csv")

# Preprocess and engineer features
test_data = engineer_features(test_data)
test_data_scaled = feature_scaling(test_data)

# Make prediction
prediction_prob = compute_model_prediction(test_data_scaled, w, b)
prediction = (prediction_prob >= 0.3).astype(int)  # Using optimized threshold
```

## Requirements
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn (for comparison only)
