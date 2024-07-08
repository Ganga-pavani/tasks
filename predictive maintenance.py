import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
train_df = pd.read_csv('/mnt/data/PM_train.txt', sep=' ', header=None)
test_df = pd.read_csv('/mnt/data/PM_test.txt', sep=' ', header=None)
truth_df = pd.read_csv('/mnt/data/PM_truth.txt', sep=' ', header=None)
train_df.drop([26, 27], axis=1, inplace=True)
test_df.drop([26, 27], axis=1, inplace=True)
truth_df.drop([1], axis=1, inplace=True)
column_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]
train_df.columns = column_names
test_df.columns = column_names
truth_df.columns = ['RUL']
rul = train_df.groupby('id')['cycle'].max().reset_index()
rul.columns = ['id', 'max_cycle']
train_df = train_df.merge(rul, on=['id'], how='left')
train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
train_df.drop('max_cycle', axis=1, inplace=True)
test_rul = test_df.groupby('id')['cycle'].max().reset_index()
test_rul.columns = ['id', 'max_cycle']
truth_df['id'] = test_rul['id']
truth_df['max_cycle'] = test_rul['max_cycle']
truth_df['RUL'] = truth_df['RUL'] + truth_df['max_cycle']
test_df = test_df.merge(test_rul, on=['id'], how='left')
test_df['RUL'] = test_df['max_cycle'] - test_df['cycle']
test_df.drop('max_cycle', axis=1, inplace=True)
scaler = StandardScaler()
scaled_train_df = train_df.copy()
scaled_test_df = test_df.copy()
scaled_train_df.iloc[:, 2:26] = scaler.fit_transform(train_df.iloc[:, 2:26])
scaled_test_df.iloc[:, 2:26] = scaler.transform(test_df.iloc[:, 2:26])
X_train = scaled_train_df.drop(['id', 'cycle', 'RUL'], axis=1)
y_train = scaled_train_df['RUL']
X_test = scaled_test_df.drop(['id', 'cycle', 'RUL'], axis=1)
y_test = scaled_test_df['RUL']
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f'Validation RMSE: {rmse}')
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, None]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))
print(f'Test RMSE: {test_rmse}')
