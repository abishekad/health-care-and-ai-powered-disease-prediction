import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('data/training.csv')

# Encode the target variable
le = LabelEncoder()
data['prognosis'] = le.fit_transform(data['prognosis'])

# Features and target
X = data.drop('prognosis', axis=1)
y = data['prognosis']

# Visualize class distribution
plt.figure(figsize=(18, 8))
sns.countplot(x=y)
plt.title("Disease Class Distribution")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Initialize models
models = {
    "Support Vector Machine": SVC(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier()
}

# Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"{model_name} Accuracy: {np.mean(scores):.2f}")

# Train and evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Test Accuracy: {acc:.2f}")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

def predict_disease(symptoms, trained_model, symptom_list, label_encoder):
    input_data = [1 if symptom in symptoms else 0 for symptom in symptom_list]
    input_df = pd.DataFrame([input_data], columns=symptom_list)
    prediction = trained_model.predict(input_df)
    disease = label_encoder.inverse_transform(prediction)
    return disease[0]

# Example usage
symptom_list = X.columns.tolist()
trained_model = models["Random Forest"]
user_symptoms = ['headache', 'fever', 'nausea']
predicted_disease = predict_disease(user_symptoms, trained_model, symptom_list, le)
print(f"Predicted Disease: {predicted_disease}")
