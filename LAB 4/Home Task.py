import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')
train_df = pd.read_csv('train2.csv')
test_df = pd.read_csv('test2.csv')
print("--- Data Inspection ---")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
def engineer_features(df):
    df = df.copy()
    df['TotalSF'] = df.get('TotalBsmtSF', 0) + df.get('1stFlrSF', 0) + df.get('2ndFlrSF', 0)
    df['TotalBath'] = df.get('FullBath', 0) + 0.5 * df.get('HalfBath', 0) + df.get('BsmtFullBath', 0) + 0.5 * df.get('BsmtHalfBath', 0)
    df['Age'] = df.get('YrSold', 2010) - df.get('YearBuilt', 2010)
    df['RemodAge'] = df.get('YrSold', 2010) - df.get('YearRemodAdd', 2010)
    df['HasPool'] = (df.get('PoolArea', 0) > 0).astype(int)
    cols_to_drop = ['Utilities', 'Id']
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])
train_clean = engineer_features(train_df)
test_clean = engineer_features(test_df)
X = train_clean.drop(columns=['SalePrice'])
y = np.log1p(train_clean['SalePrice'])
X_test_comp = test_clean.copy()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
num_imputer = SimpleImputer(strategy='median')
X[num_cols] = num_imputer.fit_transform(X[num_cols])
X_test_comp[num_cols] = num_imputer.transform(X_test_comp[num_cols])
cat_imputer = SimpleImputer(strategy='most_frequent')
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])
X_test_comp[cat_cols] = cat_imputer.transform(X_test_comp[cat_cols])
for col in cat_cols:
    le = LabelEncoder()
    full_data = pd.concat([X[col], X_test_comp[col]], axis=0).astype(str)
    le.fit(full_data)
    X[col] = le.transform(X[col].astype(str))
    X_test_comp[col] = le.transform(X_test_comp[col].astype(str))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1)
}
results = {}
print("\n--- Baseline Model Comparison (RMSLE) ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    rmse = np.sqrt(((preds - y_val) ** 2).mean())
    results[name] = rmse
    print(f"{name.ljust(18)} | Validation RMSLE: {rmse:.5f}")
print("\n--- Tuning XGBoost ---")
xgb_base = XGBRegressor(random_state=42)
param_grid = {'n_estimators': [100, 250], 'max_depth': [3, 4], 'learning_rate': [0.03, 0.05]}
grid = GridSearchCV(xgb_base, param_grid, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid.fit(X_train, y_train)
best_xgb = grid.best_estimator_
best_rmse = -grid.best_score_
print(f"Best Tuned XGBoost Validation RMSLE: {best_rmse:.5f} with params {grid.best_params_}")
final_best_model = best_xgb if best_rmse < min(results.values()) else models[min(results, key=results.get)]
print(f"\nTraining final selected model on entire dataset...")
final_best_model.fit(X, y)
test_preds_log = final_best_model.predict(X_test_comp)
test_preds = np.expm1(test_preds_log)
submission = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': test_preds})
submission.to_csv('submission2.csv', index=False)
print("Successfully generated 'submission2.csv'!")