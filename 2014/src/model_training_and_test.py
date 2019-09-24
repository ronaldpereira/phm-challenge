import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

RANDOM_SEED = 1212

train = pd.read_csv("../data/train_features.csv")

train.drop(["Time_failure", "Time_diff"], axis=1, inplace=True)

train = pd.get_dummies(train, columns=["Asset", "Reason", "Part"])

x_train, x_test, y_train, y_test = train_test_split(
    train.drop("Failure", axis=1),
    train["Failure"],
    stratify=train["Failure"],
    test_size=0.333,
    random_state=RANDOM_SEED,
)

model = RandomForestClassifier(
    n_estimators=2000, n_jobs=-1, verbose=1, random_state=RANDOM_SEED
)

model.fit(x_train, y_train)

joblib.dump(model, "random_forest_classifier_with_failure_time.joblib")

probs = model.predict_proba(x_test)

print(probs)

sc = model.score(x_test, y_test)
