# Weather Prediction Model

A machine learning project that predicts whether it will rain tomorrow based on Australian weather data using Random Forest Classification.

## Overview

This project uses historical weather data from Australia to build a classification model that predicts rain on the following day. The model achieves balanced predictions through careful data preprocessing, outlier removal, and hyperparameter tuning.

## Dataset

The project uses the `weatherAUS.csv` dataset from [kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) containing various weather measurements including temperature, humidity, wind speed, and rainfall information from different locations across Australia.

## Features

### Numeric Features
- Temperature measurements (MinTemp, MaxTemp, Temp9am, Temp3pm)
- Humidity levels (Humidity9am, Humidity3pm)
- Pressure readings (Pressure9am, Pressure3pm)
- Wind speed measurements (WindGustSpeed, WindSpeed9am, WindSpeed3pm)
- Rainfall, Evaporation, Sunshine, and Cloud coverage

### Categorical Features
- Date (Deleted)
- Location (one-hot encoded)
- Wind direction measurements (WindGustDir, WindDir9am, WindDir3pm) - one-hot encoded
- RainToday (label encoded)
- RainTomorrow (target variable, label encoded)

## Data Preprocessing

1. **Missing Value Handling**
   - Categorical features: Filled with mode
   - Numeric features: Filled with median

2. **Data Cleaning**
   - Removed Date column as it's not useful for classification

3. **Outlier Removal**
   - Applied IQR (Interquartile Range) method to all numeric features
   - Removed data points beyond 1.5 × IQR from Q1 and Q3

4. **Feature Encoding**
   - One-hot encoding for location and wind direction features
   - Label encoding for binary rain indicators

5. **Feature Scaling**
   - MinMax scaling applied to all numeric features

## Model

**Algorithm**: Random Forest Classifier

**Hyperparameters**:
- `n_estimators`: 300 trees
- `max_depth`: 6 levels
- `min_samples_leaf`: 20 samples
- `class_weight`: balanced (to handle class imbalance)

**Train-Test Split**: 80-20 split

## Evaluation Metrics

The model is evaluated using:
- F1 Score for both classes (Yes/No rain)
- Classification Report (precision, recall, f1-score)
- Confusion Matrix visualization

## Dependencies

```python
pandas
matplotlib
seaborn
scikit-learn
joblib
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure `weatherAUS.csv` is in the same directory as the script

2. Run the training script:
```bash
python main.py
```

3. The script will:
   - Load and preprocess the data
   - Train the Random Forest model
   - Display performance metrics
   - Show confusion matrix visualization
   - Save the trained model and preprocessor

## Output Files

- `randomForest.pkl`: Trained Random Forest model
- `preprocessor.pkl`: Fitted preprocessing pipeline (scaler and encoders)

## Model Performance

The model outputs:
- Detailed classification report with precision, recall, and F1-scores
- Separate F1 scores for predicting rain (Yes) and no rain (No)

- Confusion matrix showing true positives, false positives, true negatives, and false negatives

### Testing with New Data

1. Prepare your test data in `test.csv` with the same features as the training data (including RainTomorrow for validation)

2. Run the testing script:
```bash
python testing.py
```

3. The testing script will:
   - Load the saved model and preprocessor
   - Read the test data from `test.csv`
   - Apply the same preprocessing to the new data
   - Generate predictions for whether it will rain tomorrow
   - Display a confusion matrix comparing predictions vs actual values

