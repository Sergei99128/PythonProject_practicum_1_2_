
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore', category=UserWarning)

PATH = "data/parkinsons.data"
df = pd.read_csv(PATH)

y = df["status"]
X = df.drop(columns=["status", "name"])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb",     XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=500,
        use_label_encoder=False,
        n_jobs=-1,
        random_state=42
    ))
])


param_grid = {
    "xgb__n_estimators":  [200, 400, 600],
    "xgb__max_depth":     [3, 4, 5],
    "xgb__learning_rate": [0.02, 0.05, 0.1],
    "xgb__subsample":     [0.8, 1.0],
    "xgb__colsample_bytree": [0.8, 1.0],
    "xgb__gamma":         [0, 0.1],
    "xgb__reg_lambda":    [1, 5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("╒════════════════ best params ════════════════")
for k, v in grid.best_params_.items():
    print(f"{k}: {v}")
print("cross-val accuracy:", grid.best_score_)

# ===== 6. финальная оценка на test-set
y_pred = grid.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\nTest-set accuracy:", f"{acc:.2%}")
print("\nClassification report\n", classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
plt.title("Confusion matrix – Parkinson's dataset")
plt.show()

if __name__ == '__main__':
    print('is_ok')