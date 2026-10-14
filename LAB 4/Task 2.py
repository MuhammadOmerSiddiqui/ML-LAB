import os
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
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
class SimpleRandomForest:
    def __init__(self, n_trees=10, max_depth=4, random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.trees = []
        for _ in range(self.n_trees):
            indices = rng.choice(len(X), size=len(X), replace=True)
            tree = DecisionTreeClassifier(max_depth=self.max_depth, max_features="sqrt", random_state=int(rng.integers(0, 1_000_000)))
            tree.fit(X.iloc[indices], y.iloc[indices])
            self.trees.append(tree)
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([Counter(predictions[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])
scratch = SimpleRandomForest(n_trees=25, max_depth=4)
scratch.fit(X_train, y_train)
print("From-scratch Random Forest accuracy:", accuracy_score(y_test, scratch.predict(X_test)))
models = {"Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)}
for n in [10, 100, 300]:
    models[f"Random Forest (n={n})"] = RandomForestClassifier(n_estimators=n, max_depth=8, min_samples_split=4, random_state=42)
rows = []
importances = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({
        "Model": name,
        "Train Acc": accuracy_score(y_train, model.predict(X_train)),
        "Test Acc": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0)
    })
    importances[name] = pd.Series(model.feature_importances_, index=X.columns)
    print("\n" + name)
    print(classification_report(y_test, pred, zero_division=0))
table = pd.DataFrame(rows)
print("Comparison table:")
print(table.round(3).to_string(index=False))
table.to_csv(os.path.join(OUT, "task2_comparison_table.csv"), index=False)
imp_df = pd.DataFrame(importances)
print("\nFeature importance:")
print(imp_df.round(3))
imp_df.plot(kind="bar", figsize=(10, 5), title="Feature Importance: Decision Tree vs Random Forest")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task2_feature_importance.png"), dpi=120)
plt.show()
table.set_index("Model")[["Train Acc", "Test Acc"]].plot(kind="bar", figsize=(10, 5), ylim=(0, 1.05), title="Accuracy: Decision Tree vs Random Forest")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task2_accuracy.png"), dpi=120)
plt.show()
CONCLUSION = "\n".join([
    "The single Decision Tree reaches 0.952 training accuracy but only 0.667 test accuracy (F1 0.727), while the Random Forest with 10, 100 and 300 trees reaches 1.000 test accuracy. ",
    "The from-scratch forest with 25 trees scores 0.889. ",
    "Changing n_estimators from 10 to 300 did not change the test result here because the test set has only 9 students, so the difference between 10 and 300 trees cannot show up. ",
    "An ensemble can beat one tree because each tree is trained on a different bootstrap sample and a random subset of features, so the trees make different mistakes and majority voting averages those mistakes out, which lowers variance. ",
    "The importances show the same effect: the single tree relies almost only on Attendance (0.81) and ignores Assignments (0.00), whereas the forest spreads importance across Hours_Studied (0.34), Attendance (0.36) and Assignments (0.30). ",
    "The price is that the forest is less interpretable and larger than one tree, and with 9 test rows these conclusions are indicative, not statistically proven."
])
print("\nConclusion:\n" + CONCLUSION)