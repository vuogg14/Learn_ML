# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import seaborn as sns


# pre-processing data
data = pd.read_csv("Student_Performance.csv")
filter_data = data[['Hours Studied','Sleep Hours','Performance Index']]
filter_data['Performance Label'] = (filter_data['Performance Index'] > 80.0).astype(int)
filtered_data = filter_data.drop(columns=['Performance Index'])
X = filter_data[['Hours Studied','Sleep Hours']]
y = filter_data['Performance Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'C': [ 0, 0.01, 0.05, 0.1, 1, 10],  # Regularization strength
    'penalty': ['l1', 'l2'],       # Type of regularization
    'solver': ['liblinear']        # Solver supporting L1 and L2
}

# Use gridsearch to find the optimal param for accuracy 
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Evaluate the optimized model
optimized_model = grid_search.best_estimator_
y_pred_optimized = optimized_model.predict(X_test_scaled)

# Recalculate metrics for the optimized model
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
conf_matrix_optimized = confusion_matrix(y_test, y_pred_optimized)
class_report_optimized = classification_report(y_test, y_pred_optimized)

# Display the best parameters and results
print("Best Parameters:", grid_search.best_params_)
print("Optimized Accuracy:", accuracy_optimized)
print("\nOptimized Classification Report:\n", class_report_optimized)

# Confusion matrix heatmap for the optimized model
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_optimized, annot=True, fmt="d", cmap="Greens", xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
plt.title("Confusion Matrix (Optimized Model)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()