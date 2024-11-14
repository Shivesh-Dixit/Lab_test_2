from sklearn.datasets import load_wine  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.preprocessing import StandardScaler  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report  
import pandas as pd  
import matplotlib.pyplot as plt  
import numpy as np


wine_data = load_wine()
wind_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
wind_df['target'] = wine_data.target
print(wind_df.head(10))  

print(f"Number of samples: {wind_df.shape[0]}")  
print(f"Number of features: {wind_df.shape[1] - 1}")  
print(f"Number of classes: {len(wine_data.target_names)}") 

X_train, X_test, y_train, y_test = train_test_split(wine_data.data, wine_data.target, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  

wind_df.hist(bins=20, figsize=(20, 30))
plt.suptitle("Feature distribution before scaling")
plt.show()

pd.DataFrame(X_train_scaled, columns=wine_data.feature_names).hist(bins=20, figsize=(20, 30))
plt.suptitle("Feature distribution after scaling")
plt.show()

dt_model = DecisionTreeClassifier()

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(dt_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print("Best hyperparameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, y_pred))



