import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from xgboost import XGBClassifier
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
df = pd.DataFrame({
    "Hours_Studied": [1, 2, 3, 5, 2, 3, 1, 7, 8, 6, 6, 5, 8, 1, 8, 1, 9, 1, 8, 2, 7, 2, 7, 5, 5, 4, 8, 4, 1, 8],
    "Attendance":    [60, 40, 52, 71, 59, 69, 51, 80, 81, 73, 71, 75, 82, 52, 96, 57, 82, 44, 87, 44, 57, 40, 80, 63, 61, 67, 69, 83, 45, 77],
    "Assignments":   [5, 0, 2, 5, 0, 2, 2, 7, 8, 8, 10, 8, 4, 4, 10, 2, 7, 0, 8, 0, 9, 4, 7, 3, 8, 5, 10, 3, 6, 8],
    "Pass":          [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1]
})
X = df[["Hours_Studied", "Attendance", "Assignments"]]
y = df["Pass"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
def evaluate(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "Model": name,
        "Train Acc": accuracy_score(y_train, model.predict(X_train)),
        "Test Acc": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "Log Loss": log_loss(y_test, model.predict_proba(X_test))
    }
xgb_configs = [(2, 0.05), (3, 0.1), (4, 0.3)]
xgb_rows = []
xgb_models = {}
for depth, lr in xgb_configs:
    name = f"XGBoost (depth={depth}, lr={lr})"
    model = XGBClassifier(n_estimators=200, max_depth=depth, learning_rate=lr, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)
    xgb_rows.append(evaluate(name, model))
    xgb_models[name] = model
xgb_table = pd.DataFrame(xgb_rows)
print("XGBoost tuning results:")
print(xgb_table.round(3).to_string(index=False))
best_xgb_name = xgb_table.sort_values(["Test Acc", "F1", "Log Loss"], ascending=[False, False, True]).iloc[0]["Model"]
print("\nBest XGBoost configuration:", best_xgb_name)
xgb_importance = pd.Series(xgb_models[best_xgb_name].feature_importances_, index=X.columns).sort_values(ascending=False)
xgb_importance.plot(kind="bar", figsize=(8, 4), title="XGBoost Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task4_xgb_feature_importance.png"), dpi=120)
plt.show()
rows = [
    evaluate("Decision Tree", DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)),
    evaluate("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=4, random_state=42)),
    evaluate("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)),
    xgb_table[xgb_table["Model"] == best_xgb_name].iloc[0].to_dict()
]
rows[-1]["Model"] = "XGBoost (best: " + best_xgb_name.split("(")[1]
comparison = pd.DataFrame(rows)
print("\nModel comparison table:")
print(comparison.round(3).to_string(index=False))
comparison.to_csv(os.path.join(OUT, "task4_model_comparison.csv"), index=False)
xgb_table.to_csv(os.path.join(OUT, "task4_xgboost_tuning.csv"), index=False)
comparison.set_index("Model")[["Train Acc", "Test Acc", "Precision", "Recall", "F1"]].plot(kind="bar", figsize=(11, 5), ylim=(0, 1.05), title="Model Comparison on the Same Split")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task4_model_comparison.png"), dpi=120)
plt.show()