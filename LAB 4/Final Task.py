# The lab says "provided/specified dataset" but no CSV was attached, so if smartphone_addiction.csv is not found next to this
# script, a synthetic dataset with the exact columns listed in section 8.2 of the manual is generated.
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
CSV = os.path.join(HERE, "smartphone_addiction.csv")
rng = np.random.default_rng(42)
def make_synthetic(n=1500):
    age = rng.integers(15, 55, n)
    social = np.clip(rng.normal(2.5, 1.4, n), 0, 9)
    gaming = np.clip(rng.exponential(1.0, n), 0, 8)
    work_study = np.clip(rng.normal(5, 2, n), 0, 12)
    other = np.clip(rng.normal(1.5, 0.8, n), 0, 5)
    daily = social + gaming + other + 0.3 * work_study
    sleep = np.clip(8 - 0.25 * daily + rng.normal(0, 0.9, n), 3, 10)
    notifications = np.clip(30 + 12 * daily + rng.normal(0, 25, n), 0, 400).astype(int)
    opens = np.clip(20 + 9 * daily + rng.normal(0, 20, n), 0, 300).astype(int)
    weekend = np.clip(daily + rng.normal(1.0, 1.0, n), 0, 16)
    gender = rng.choice(["Male", "Female", "Other"], n, p=[0.48, 0.48, 0.04])
    stress_score = 0.35 * daily - 0.3 * sleep + rng.normal(0, 1.2, n)
    stress = np.where(stress_score > 0.2, "High", np.where(stress_score > -0.6, "Medium", "Low"))
    risk = 0.55 * daily + 0.02 * opens + 0.015 * notifications - 0.45 * sleep + 0.35 * (stress == "High") + 0.4 * (stress == "Medium") - 0.04 * (age - 30) + rng.normal(0, 1.1, n)
    addicted = (risk > np.quantile(risk, 0.62)).astype(int)
    impact_score = 0.5 * social + 0.4 * gaming - 0.3 * work_study / 3 + 0.8 * addicted + rng.normal(0, 1, n)
    impact = np.where(impact_score > 2.6, "Severe", np.where(impact_score > 1.6, "Moderate", "None"))
    d = pd.DataFrame({
        "age": age, "daily_screen_time_hours": daily.round(2), "social_media_hours": social.round(2), "gaming_hours": gaming.round(2),
        "work_study_hours": work_study.round(2), "sleep_hours": sleep.round(2), "notifications_per_day": notifications,
        "app_opens_per_day": opens, "weekend_screen_time": weekend.round(2), "gender": gender, "stress_level": stress,
        "academic_work_impact": impact, "addicted": addicted
    })
    for col in ["sleep_hours", "social_media_hours", "stress_level", "age"]:
        d.loc[rng.random(n) < 0.03, col] = np.nan
    return d
if os.path.exists(CSV):
    df = pd.read_csv(CSV)
    print("Loaded real dataset:", CSV)
else:
    df = make_synthetic()
    df.to_csv(os.path.join(OUT, "smartphone_addiction_synthetic.csv"), index=False)
    print("smartphone_addiction.csv not found -> using synthetic dataset (saved to outputs/)")
TARGET = "addicted"
numeric_cols = ["age", "daily_screen_time_hours", "social_media_hours", "gaming_hours", "work_study_hours", "sleep_hours", "notifications_per_day", "app_opens_per_day", "weekend_screen_time"]
print("\nShape:", df.shape)
print("Columns:", list(df.columns))
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])
print("\nDuplicate rows:", int(df.duplicated().sum()))
print("\nTarget distribution:\n", df[TARGET].value_counts(normalize=True).round(3))
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
df[TARGET].value_counts().sort_index().plot(kind="bar", ax=axes[0, 0], color=["tab:green", "tab:red"], title="Target distribution (0 = not addicted, 1 = addicted)")
sns.histplot(data=df, x="daily_screen_time_hours", hue=TARGET, kde=True, ax=axes[0, 1])
axes[0, 1].set_title("Daily screen time by class")
sns.boxplot(data=df, x=TARGET, y="sleep_hours", ax=axes[1, 0])
axes[1, 0].set_title("Sleep hours by class")
sns.heatmap(df[numeric_cols + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1], annot_kws={"size": 7})
axes[1, 1].set_title("Correlation heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "final_eda.png"), dpi=120)
plt.show()
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)
print("\nSplit sizes -> train:", len(X_train), "| validation:", len(X_val), "| test:", len(X_test))
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("stress", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ord", OrdinalEncoder(categories=[["Low", "Medium", "High"]]))]), ["stress_level"]),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), ["gender", "academic_work_impact"])
])
preprocess.fit(X_train)
feature_names = [n.split("__")[-1] for n in preprocess.get_feature_names_out()]
Xtr, Xva, Xte = [pd.DataFrame(preprocess.transform(part), columns=feature_names) for part in (X_train, X_val, X_test)]
models = {
    "Decision Tree": DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, max_features="sqrt", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=15, max_depth=5, random_state=42, verbosity=-1)
}
def metrics(model, Xs, ys):
    p = model.predict(Xs)
    return {"Accuracy": accuracy_score(ys, p), "Precision": precision_score(ys, p, zero_division=0), "Recall": recall_score(ys, p, zero_division=0), "F1": f1_score(ys, p, zero_division=0)}
val_rows, test_rows = [], []
for name, model in models.items():
    model.fit(Xtr, y_train)
    val_rows.append({"Model": name, **metrics(model, Xva, y_val)})
    test_rows.append({"Model": name, **metrics(model, Xte, y_test)})
val_table = pd.DataFrame(val_rows).set_index("Model")
test_table = pd.DataFrame(test_rows).set_index("Model")
print("\nValidation comparison:")
print(val_table.round(4))
print("\nTest comparison:")
print(test_table.round(4))
val_table.to_csv(os.path.join(OUT, "final_validation_comparison.csv"))
test_table.to_csv(os.path.join(OUT, "final_test_comparison.csv"))
val_table.plot(kind="bar", figsize=(11, 5), ylim=(0, 1.05), title="Validation metrics")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "final_model_comparison.png"), dpi=120)
plt.show()
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for ax, (name, model) in zip(axes, models.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_test, model.predict(Xte)), display_labels=["Not addicted", "Addicted"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "final_confusion_matrices.png"), dpi=120)
plt.show()
final_name = val_table["F1"].idxmax()
final_model = models[final_name]
print("\nFinal model (best validation F1):", final_name)
print("Its test metrics:", test_table.loc[final_name].round(4).to_dict())
importance = pd.Series(final_model.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\nTop features:\n", importance.head(8).round(4))
importance.head(12).sort_values().plot(kind="barh", figsize=(8, 5), title=f"Feature importance ({final_name})")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "final_feature_importance.png"), dpi=120)
plt.show()
unseen = X_test.copy()
predictions = pd.DataFrame({"row": unseen.index, "predicted_addicted": final_model.predict(Xte), "probability": final_model.predict_proba(Xte)[:, 1].round(4)})
predictions.to_csv(os.path.join(OUT, "final_predictions.csv"), index=False)
print("\nSample predictions:\n", predictions.head())