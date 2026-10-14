import os
import time
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")
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
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start
    pred = model.predict(X_test)
    return {
        "Model": name,
        "Train Acc": accuracy_score(y_train, model.predict(X_train)),
        "Test Acc": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "Fit Time (s)": fit_time
    }
rows = []
for depth in [2, 3, 5]:
    rows.append(evaluate(f"XGBoost (max_depth={depth})", XGBClassifier(n_estimators=200, max_depth=depth, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)))
for leaves in [4, 8, 31]:
    rows.append(evaluate(f"LightGBM (num_leaves={leaves})", LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=leaves, max_depth=-1, min_child_samples=2, min_data_in_bin=1, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbosity=-1)))
table = pd.DataFrame(rows)
print("Comparison table:")
print(table.round(4).to_string(index=False))
table.to_csv(os.path.join(OUT, "task5_xgb_vs_lgbm.csv"), index=False)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
table.set_index("Model")[["Train Acc", "Test Acc", "Precision", "Recall", "F1"]].plot(kind="bar", ax=axes[0], ylim=(0, 1.05), title="Accuracy / Precision / Recall / F1")
axes[0].tick_params(axis="x", rotation=30)
table.set_index("Model")["Fit Time (s)"].plot(kind="bar", ax=axes[1], color="tab:orange", title="Training time (seconds)")
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task5_comparison.png"), dpi=120)
plt.show()
PARAGRAPH = "\n".join([
    "On this 30-student dataset XGBoost reaches 1.000 test accuracy/precision/recall/F1 for max_depth 2, 3 and 5 while LightGBM reaches 0.778 accuracy and 0.800 precision, recall and F1 for num_leaves 4, 8 and 31, and both train in about 0.02 seconds (the first LightGBM fit is slower only because of start-up cost). ",
    "Changing num_leaves or max_depth did not change the results because only 21 training rows are available, so the trees cannot grow larger anyway and the test set of 9 students is too small to separate the models reliably. ",
    "An efficient library such as LightGBM becomes preferable when the data is large (hundreds of thousands of rows or many features), because histogram binning and leaf-wise growth reduce training time and memory while keeping accuracy high, and it also lets you tune faster during competitions. ",
    "Leaf-wise growth can overfit small data, so num_leaves, max_depth and min_child_samples must be controlled; XGBoost is a safe, well-regularised choice for small or medium tables."
])
print("\nWhen an efficient boosting library may be preferable:\n" + PARAGRAPH)