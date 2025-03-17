# personalized-healthcare-recommendations

## Overview
This project is a **Healthcare Recommendation System** that analyzes patient data and predicts recommendations based on historical records. It leverages **Machine Learning (ML)** techniques to classify whether a patient requires a healthcare recommendation or not.

## Features
- **Data Preprocessing**: Handles missing values, normalizes numerical features, and encodes categorical variables.
- **Machine Learning Pipeline**: Uses **Random Forest Classifier** for prediction.
- **Data Visualization**: Includes distribution plots, scatter plots, and correlation heatmaps.
- **Performance Evaluation**: Provides classification reports and confusion matrices for model evaluation.

## Dataset
The dataset used is `blood.csv`, which contains patient records with the following key features:

- **recency**: Time since the last visit.
- **frequency**: Number of past visits.
- **monetary**: Donation amount or associated cost.
- **time**: Duration of follow-up.
- **class**: Categorical classification of patients.
- **recommendation** (if available): Target variable for prediction; otherwise, it is generated based on specific conditions (`recency > 6` & `monetary > 500`).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Ensure the `blood.csv` dataset is in the project directory.
2. Run the script:
   ```bash
   python phrs.py
   ```
3. View the model evaluation metrics and generated visualizations.

## Machine Learning Pipeline
### 1. Data Preprocessing
   - Standardizes numerical features (`recency`, `frequency`, `monetary`, `time`).
   - Encodes categorical variables (`class`).
### 2. Model Training
   - Splits data into training and testing sets.
   - Uses **Random Forest Classifier** for prediction.
### 3. Evaluation & Visualization
   - Generates a **confusion matrix** and **classification report**.
   - Produces various plots like **pair plots, scatter plots, histograms**, and **heatmaps** to understand feature relationships.

## Visualizations
- **Recommendation Distribution**: Displays the count of different recommendation types.
- **Feature Correlation Heatmap**: Shows the relationship between numerical features.
- **Scatter Plots**: Analyzes recency vs. frequency.
- **Histograms**: Visualizes feature distributions.
- **ROC Curve**: Evaluates the model's classification performance (if applicable).

## Dependencies
- `pandas`
- `seaborn`
- `matplotlib`
- `sklearn`

## Future Improvements
- Implement hyperparameter tuning for better model performance.
- Add more complex ML models like **XGBoost** or **Neural Networks**.
- Optimize feature selection and engineering.

## Author
**Harsha Priya Putta** -https://github.com/HarshaPriya19

## License
This project is licensed under the **MIT License** - see the LICENSE file for details.

