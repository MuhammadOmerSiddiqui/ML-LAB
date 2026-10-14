import os
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
if os.path.exists("spaceship-titanic.zip") and not os.path.exists("train.csv"):
    with zipfile.ZipFile("spaceship-titanic.zip", "r") as zip_ref:
        zip_ref.extractall(".")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
def engineer_features(df):
    df = df.copy()
    df[['Group', 'Member']] = df['PassengerId'].str.split('_', expand=True)
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Num'] = pd.to_numeric(df['Num'], errors='coerce')
    exp_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpending'] = df[exp_cols].sum(axis=1)
    df['NoSpending'] = (df['TotalSpending'] == 0).astype(int)
    df = df.drop(columns=['PassengerId', 'Cabin', 'Name', 'Member'])
    return df
train_clean = engineer_features(train_df)
test_clean = engineer_features(test_df)
X = train_clean.drop(columns=['Transported'])
y = train_clean['Transported'].astype(int)
X_test_comp = test_clean.copy()
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
num_imputer = SimpleImputer(strategy='median')
X[num_cols] = num_imputer.fit_transform(X[num_cols])
X_test_comp[num_cols] = num_imputer.transform(X_test_comp[num_cols])
cat_imputer = SimpleImputer(strategy='most_frequent')
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols].astype(str))
X_test_comp[cat_cols] = cat_imputer.transform(X_test_comp[cat_cols].astype(str))
for col in cat_cols:
    le = LabelEncoder()
    full_data = pd.concat([X[col], X_test_comp[col]], axis=0)
    le.fit(full_data)
    X[col] = le.transform(X[col])
    X_test_comp[col] = le.transform(X_test_comp[col])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
models = {
    "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, max_depth=-1, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1)
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    results[name] = acc
    print(f"Model: {name} | Validation Accuracy: {acc:.4f}")
best_model_name = max(results, key=results.get)
print(f"\nBest Performing Model: {best_model_name} with Accuracy: {results[best_model_name]:.4f}\n")
best_model = models[best_model_name]
best_model.fit(X, y)
test_preds = best_model.predict(X_test_comp)
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Transported": test_preds.astype(bool)
})
submission.to_csv("submission.csv", index=False)
print("Successfully generated 'submission.csv'!")