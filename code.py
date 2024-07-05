import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load the dataset
file_path = 'https://github.com/Ganga-pavani/tasks/blob/main/Churn-Data.csv'
df = pd.read_csv(file_path)

# Display basic information about the dataset
print(df.info())
print(df.describe())
print(df.head())
print(df.isnull().sum())

# Handling missing values
df.fillna(method='ffill', inplace=True)

# Convert categorical variables to numeric
df = pd.get_dummies(df, drop_first=True)

# Feature Engineering
# Example: Creating a feature that captures the total usage
# Adjust these column names as per your dataset
if 'DayMins' in df.columns and 'EveMins' in df.columns and 'NightMins' in df.columns and 'IntlMins' in df.columns:
    df['TotalUsage'] = df['DayMins'] + df['EveMins'] + df['NightMins'] + df['IntlMins']

# Drop less useful columns if necessary
# Adjust these column names as per your dataset
if 'PhoneNumber' in df.columns:
    df.drop(['PhoneNumber'], axis=1, inplace=True)

# Define the feature matrix and target vector
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model Training and Evaluation
# Initialize the model with class weights to handle imbalance
model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'F1-Score: {f1}')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Optimize and Validate the Model using Grid Search
# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, None]
}

# Initialize Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=5)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters
print(grid_search.best_params_)

# Predict with the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Evaluate the best model
best_accuracy = accuracy_score(y_test, y_pred_best)
best_f1 = f1_score(y_test, y_pred_best)

print(f'Best Accuracy: {best_accuracy}')
print(f'Best F1-Score: {best_f1}')
print(classification_report(y_test, y_pred_best))
print(confusion_matrix(y_test, y_pred_best))
