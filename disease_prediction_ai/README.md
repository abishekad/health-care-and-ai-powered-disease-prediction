# AI-Powered Disease Prediction System

This project implements a machine learning-based system to predict diseases based on patient-reported symptoms.

## Dataset

- Source: [Kaggle - Disease Prediction Using Machine Learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)

## Features

- Utilizes Support Vector Machine, Naive Bayes, and Random Forest classifiers.
- Implements Stratified K-Fold Cross-Validation.
- Provides a function to predict diseases based on user-input symptoms.

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the `disease_prediction.py` script:

```bash
python disease_prediction.py
```

## Example

```python
user_symptoms = ['headache', 'fever', 'nausea']
predicted_disease = predict_disease(user_symptoms, trained_model, symptom_list, le)
print(f"Predicted Disease: {predicted_disease}")
```
