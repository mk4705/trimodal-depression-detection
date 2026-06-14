"""
Video Emotion Recognition using Hierarchical SVM
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("final_dataset.csv")


def add_labels(df):
    label_map = {
        1: "neutral",
        2: "happy",
        3: "sad",
        4: "angry",
        5: "fear",
        6: "disgust",
        7: "surprise",
    }

    df["emotion"] = df["label"].map(label_map)

    df["level1"] = df["emotion"].apply(
        lambda x: "neutral" if x == "neutral" else "non-neutral"
    )

    df["level2"] = df["emotion"].apply(
        lambda x: "positive"
        if x in ["happy", "surprise"]
        else ("negative" if x in ["sad", "disgust", "angry", "fear"] else "neutral")
    )

    return df


df = add_labels(df)

train_df = df[df["filename"].str.startswith("train")]
test_df = df[df["filename"].str.startswith("test")]

feature_cols = [
    c
    for c in df.columns
    if c not in ["filename", "label", "emotion", "level1", "level2"]
]

X_train = train_df[feature_cols]
y_train = train_df["emotion"]

X_test = test_df[feature_cols]
y_test = test_df["emotion"]

X_train = X_train[~X_train.isna().all(axis=1)]
X_test = X_test[~X_test.isna().all(axis=1)]

scaler = StandardScaler()
imputer = SimpleImputer(strategy="median")

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

boruta = BorutaPy(
    rf,
    n_estimators="auto",
    random_state=42
)

boruta.fit(X_train, y_train.values)

X_train = boruta.transform(X_train)
X_test = boruta.transform(X_test)

y_l1 = y_train.apply(
    lambda x: "neutral" if x == "neutral" else "non-neutral"
)

svm_l1 = SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced"
).fit(X_train, y_l1)

mask = y_train != "neutral"

X_l2 = X_train[mask]

y_l2 = y_train[mask].apply(
    lambda x: "positive"
    if x in ["happy", "surprise"]
    else "negative"
)

svm_l2 = SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced"
).fit(X_l2, y_l2)

svm_pos = SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced"
).fit(
    X_train[y_train.isin(["happy", "surprise"])],
    y_train[y_train.isin(["happy", "surprise"])]
)

svm_sd = SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced"
).fit(
    X_train[y_train.isin(["sad", "disgust"])],
    y_train[y_train.isin(["sad", "disgust"])]
)

svm_af = SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced"
).fit(
    X_train[y_train.isin(["angry", "fear"])],
    y_train[y_train.isin(["angry", "fear"])]
)


def predict_sample(x):
    l1 = svm_l1.predict(x.reshape(1, -1))[0]

    if l1 == "neutral":
        return "neutral"

    l2 = svm_l2.predict(x.reshape(1, -1))[0]

    if l2 == "positive":
        return svm_pos.predict(x.reshape(1, -1))[0]

    prob_sd = svm_sd.predict_proba(x.reshape(1, -1))[0]
    prob_af = svm_af.predict_proba(x.reshape(1, -1))[0]

    if max(prob_sd) > max(prob_af):
        return svm_sd.classes_[np.argmax(prob_sd)]

    return svm_af.classes_[np.argmax(prob_af)]


y_pred = [predict_sample(x) for x in X_test]

models = {
    "scaler": scaler,
    "imputer": imputer,
    "boruta": boruta,
    "svm_l1": svm_l1,
    "svm_l2": svm_l2,
    "svm_pos": svm_pos,
    "svm_sd": svm_sd,
    "svm_af": svm_af,
}

joblib.dump(models, "hierarchical_model.pkl")
