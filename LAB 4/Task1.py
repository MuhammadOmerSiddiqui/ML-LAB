import math
import os
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)
df = pd.DataFrame({
    "Hours_Studied": [1, 2, 3, 5, 2, 3, 1, 7, 8, 6, 6, 5, 8, 1, 8, 1, 9, 1, 8, 2, 7, 2, 7, 5, 5, 4, 8, 4, 1, 8],
    "Attendance":    [60, 40, 52, 71, 59, 69, 51, 80, 81, 73, 71, 75, 82, 52, 96, 57, 82, 44, 87, 44, 57, 40, 80, 63, 61, 67, 69, 83, 45, 77],
    "Assignments":   [5, 0, 2, 5, 0, 2, 2, 7, 8, 8, 10, 8, 4, 4, 10, 2, 7, 0, 8, 0, 9, 4, 7, 3, 8, 5, 10, 3, 6, 8],
    "Pass":          [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1]
})
print(df)
print("\nClass balance:\n", df["Pass"].value_counts())
def entropy(y):
    counts = Counter(y)
    n = len(y)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())
def information_gain(y, left_y, right_y):
    n = len(y)
    weighted = (len(left_y) / n) * entropy(left_y) + (len(right_y) / n) * entropy(right_y)
    return entropy(y) - weighted
def best_split(X, y):
    best, best_gain = None, -1
    for feature in X.columns:
        for value in sorted(X[feature].unique()):
            left = y[X[feature] <= value]
            right = y[X[feature] > value]
            if len(left) == 0 or len(right) == 0:
                continue
            gain = information_gain(y, left, right)
            if gain > best_gain:
                best_gain = gain
                best = (feature, value)
    return best, best_gain
features = ["Hours_Studied", "Attendance", "Assignments"]
X = df[features]
y = df["Pass"]
split, gain = best_split(X, y)
print("\nFrom-scratch best root split:", split)
print("Information Gain:", round(gain, 4))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
rows = []
models = {}
for criterion in ["gini", "entropy"]:
    for depth in [2, 3, 5]:
        model = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        models[(criterion, depth)] = model
        rows.append({
            "criterion": criterion,
            "max_depth": depth,
            "train_acc": accuracy_score(y_train, model.predict(X_train)),
            "test_acc": accuracy_score(y_test, model.predict(X_test)),
            "actual_depth": model.get_depth(),
            "n_leaves": model.get_n_leaves()
        })
results = pd.DataFrame(rows)
print("\nAccuracy comparison:")
print(results.round(3).to_string(index=False))
results.to_csv(os.path.join(OUT, "task1_accuracy_comparison.csv"), index=False)
best_row = results.sort_values(["test_acc", "train_acc"], ascending=False).iloc[0]
best_model = models[(best_row["criterion"], int(best_row["max_depth"]))]
print("\nBest setting:", best_row["criterion"], "max_depth =", int(best_row["max_depth"]))
print(classification_report(y_test, best_model.predict(X_test), zero_division=0))
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
for ax, ((criterion, depth), model) in zip(axes.ravel(), models.items()):
    plot_tree(model, feature_names=features, class_names=["Fail", "Pass"], filled=True, ax=ax, fontsize=7)
    ax.set_title(f"criterion={criterion}, max_depth={depth}")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task1_all_trees.png"), dpi=120)
plt.show()
plt.figure(figsize=(12, 7))
plot_tree(best_model, feature_names=features, class_names=["Fail", "Pass"], filled=True)
plt.title("Best Decision Tree")
plt.savefig(os.path.join(OUT, "task1_best_tree.png"), dpi=120)
plt.show()
pivot_train = results.pivot(index="max_depth", columns="criterion", values="train_acc")
pivot_test = results.pivot(index="max_depth", columns="criterion", values="test_acc")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
pivot_train.plot(kind="bar", ax=axes[0], ylim=(0, 1.05), title="Train accuracy")
pivot_test.plot(kind="bar", ax=axes[1], ylim=(0, 1.05), title="Test accuracy")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task1_accuracy_bars.png"), dpi=120)
plt.show()
OBSERVATIONS = "\n".join([
    "1. The dataset has 30 students (17 Pass, 13 Fail), split 70/30 into 21 training and 9 test students. ",
    "2. Training accuracy rises with depth: gini goes 0.905 -> 0.952 -> 1.000 and entropy goes 0.857 -> 0.952 -> 1.000 for max_depth 2, 3 and 5. ",
    "3. max_depth=5 never actually uses 5 levels (the fitted depth is 4) because the tree stops as soon as every leaf is pure. ",
    "4. Test accuracy does not follow training accuracy: gini scores 1.000, 1.000 and 0.889, and entropy scores 0.778, 0.667 and 0.667 for depth 2, 3 and 5. ",
    "5. The growing gap between training and test accuracy at depth 5 is the sign of overfitting, because the deep tree memorises noisy students instead of the general pattern. ",
    "6. Very shallow trees (depth 2) underfit slightly (training accuracy 0.857-0.905) but generalise as well as or better than deeper trees on this data. ",
    "7. Gini and entropy pick different splits on such a small dataset, so their accuracies differ, but with only 9 test students one mistake equals 11 percentage points, so that difference is not reliable. ",
    "8. Conclusion: a depth of 2-3 is the best bias/variance trade-off here, and limiting depth (or pruning) is the main way to control a single decision tree."
])
print("\nObservations:\n" + OBSERVATIONS)
