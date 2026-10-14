import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
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
X_demo = np.array([[1], [2], [3], [4], [5], [6]])
y_demo = np.array([2, 4, 5, 8, 10, 12])
prediction = np.full(len(y_demo), y_demo.mean())
for stage in range(3):
    residual = y_demo - prediction
    learner = DecisionTreeRegressor(max_depth=1, random_state=stage)
    learner.fit(X_demo, residual)
    prediction = prediction + 0.3 * learner.predict(X_demo)
    print("Stage:", stage + 1, "Predictions:", np.round(prediction, 2))
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
rows = []
for lr in [0.01, 0.05, 0.1]:
    for n in [50, 100, 200]:
        gb = GradientBoostingClassifier(n_estimators=n, learning_rate=lr, max_depth=3, random_state=42)
        val = cross_val_score(gb, X_train, y_train, cv=cv, scoring="accuracy").mean()
        gb.fit(X_train, y_train)
        rows.append({
            "learning_rate": lr,
            "n_estimators": n,
            "train_acc": accuracy_score(y_train, gb.predict(X_train)),
            "val_acc (CV)": val,
            "test_acc": accuracy_score(y_test, gb.predict(X_test))
        })
results = pd.DataFrame(rows)
print("\nParameter-result table:")
print(results.round(3).to_string(index=False))
results.to_csv(os.path.join(OUT, "task3_parameter_results.csv"), index=False)
best = results.sort_values(["val_acc (CV)", "test_acc"], ascending=False).iloc[0]
print("\nBest validation result: learning_rate =", best["learning_rate"], ", n_estimators =", int(best["n_estimators"]), ", val_acc =", round(best["val_acc (CV)"], 3))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, col in zip(axes, ["val_acc (CV)", "test_acc"]):
    grid = results.pivot(index="learning_rate", columns="n_estimators", values=col)
    im = ax.imshow(grid.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns)
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index)
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("learning_rate")
    ax.set_title(col)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid.values[i, j]:.2f}", ha="center", va="center", color="white")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "task3_heatmap.png"), dpi=120)
plt.show()
plt.figure(figsize=(8, 5))
for lr in [0.01, 0.05, 0.1]:
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=lr, max_depth=3, random_state=42).fit(X_train, y_train)
    curve = [accuracy_score(y_test, p) for p in gb.staged_predict(X_test)]
    plt.plot(range(1, 201), curve, label=f"learning_rate={lr}")
plt.xlabel("Number of trees")
plt.ylabel("Test accuracy")
plt.title("Learning rate vs number of estimators")
plt.legend()
plt.savefig(os.path.join(OUT, "task3_staged_curve.png"), dpi=120)
plt.show()
EXPLANATION = "\n".join([
    "The best 4-fold validation accuracy is 0.717, reached by several settings (learning_rate=0.01 with 200 trees, and learning_rate 0.05 or 0.1 with some tree counts), and the script reports learning_rate=0.01, n_estimators=200 as the first of the ties. ",
    "Learning rate and number of estimators work against each other: each tree only contributes learning_rate times its correction, so a small learning rate needs many more trees. ",
    "With learning_rate=0.01, 50 or 100 trees are not enough (training accuracy 0.952, validation 0.667) and 200 trees are needed to reach training accuracy 1.000. ",
    "With learning_rate 0.05 or 0.1 the model already reaches training accuracy 1.000 with 50 trees, so it learns faster and costs less compute but can overfit sooner. ",
    "The trade-off is therefore: a small learning rate with many trees is slower but usually smoother and more robust, while a large learning rate with few trees is fast but riskier. ",
    "With only 21 training rows each validation fold holds about 5 students and the test set has 9, so differences such as 0.617 versus 0.717 are within noise and should not be over-interpreted."
])
print("\nExplanation:\n" + EXPLANATION)