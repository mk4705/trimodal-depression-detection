import pandas as pd
import shap
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD DATA
# =========================

text_df = pd.read_csv("/kaggle/input/datasets/robinsavio/finalfiles/MiniProject/text_csv.csv")
audio_df = pd.read_csv("/kaggle/input/datasets/robinsavio/finalfiles/MiniProject/audio_csv.csv")
video_df = pd.read_csv("/kaggle/input/datasets/robinsavio/finalfiles/MiniProject/finalVideo.csv")

train_labels = pd.read_csv(
    "/kaggle/input/datasets/robinsavio/finalfiles/MiniProject/train_split_Depression_AVEC2017.csv"
)

test_labels = pd.read_csv(
    "/kaggle/input/datasets/robinsavio/finalfiles/MiniProject/full_test_split.csv"
)

# =========================
# COLUMN RENAMING
# =========================

train_labels = train_labels.rename(columns={
    "Participant_ID": "participant_id",
    "PHQ8_Binary": "label",
    "PHQ8_Score": "score",
    "Gender": "gender"
})

test_labels = test_labels.rename(columns={
    "Participant_ID": "participant_id",
    "PHQ8_Binary": "label",
    "PHQ8_Score": "score",
    "Gender": "gender"
})

train_labels = train_labels[
    ["participant_id", "label", "score", "gender"]
]

test_labels = test_labels[
    ["participant_id", "label", "score", "gender"]
]

labels_df = pd.concat(
    [train_labels, test_labels],
    ignore_index=True
)

# =========================
# ID CLEANING
# =========================

for df_ in [text_df, audio_df, video_df, labels_df]:
    df_["participant_id"] = (
        df_["participant_id"]
        .astype(str)
        .str.strip()
        .astype(int)
    )

emotions = [
    "sadness",
    "joy",
    "anger",
    "neutral",
    "disgust",
    "surprise",
    "fear"
]

cols = ["participant_id"] + emotions

text_df = text_df[cols]
audio_df = audio_df[cols]
video_df = video_df[cols]

# =========================
# COMMON IDS
# =========================

common_ids = (
    set(text_df["participant_id"]) &
    set(audio_df["participant_id"]) &
    set(video_df["participant_id"])
)

text_df = text_df[text_df["participant_id"].isin(common_ids)]
audio_df = audio_df[audio_df["participant_id"].isin(common_ids)]
video_df = video_df[video_df["participant_id"].isin(common_ids)]

text_df = text_df.set_index("participant_id")
audio_df = audio_df.set_index("participant_id")
video_df = video_df.set_index("participant_id")

df = text_df.join(audio_df, rsuffix="_audio")
df = df.join(video_df, rsuffix="_video")

df = df.reset_index()

df = df.merge(labels_df, on="participant_id")

print("Final dataset size:", len(df))

# =========================
# THRESHOLD GENERATION
# =========================

train_ids = set(train_labels["participant_id"])
test_ids = set(test_labels["participant_id"])

train_df = df[df["participant_id"].isin(train_ids)]

def compute_thresholds(df, prefix, q=0.75):
    return {
        emo: df[f"{emo}_{prefix}"].quantile(q)
        for emo in emotions
    }

th_text = compute_thresholds(train_df, "text")
th_audio = compute_thresholds(train_df, "audio")
th_video = compute_thresholds(train_df, "video")

# =========================
# CREATE FLAGS
# =========================

def create_flags(df, prefix, thresholds):

    for emo in emotions:
        df[f"{emo}_{prefix}_flag"] = (
            df[f"{emo}_{prefix}"] > thresholds[emo]
        ).astype(int)

    return df

df = create_flags(df, "text", th_text)
df = create_flags(df, "audio", th_audio)
df = create_flags(df, "video", th_video)

for emo in emotions:

    df[f"{emo}_agreement"] = (
        df[f"{emo}_text_flag"] +
        df[f"{emo}_audio_flag"] +
        df[f"{emo}_video_flag"]
    ) >= 2

    df[f"{emo}_agreement"] = (
        df[f"{emo}_agreement"].astype(int)
    )

# =========================
# FEATURES
# =========================

flag_cols = []

for emo in emotions:
    flag_cols += [
        f"{emo}_text_flag",
        f"{emo}_audio_flag",
        f"{emo}_video_flag"
    ]

agreement_cols = [
    f"{emo}_agreement"
    for emo in emotions
]

X = df[flag_cols + agreement_cols].copy()
X["gender"] = df["gender"]

y = df["label"]

X_train = X[df["participant_id"].isin(train_ids)]
X_test = X[df["participant_id"].isin(test_ids)]

y_train = y[df["participant_id"].isin(train_ids)]
y_test = y[df["participant_id"].isin(test_ids)]

print("Train:", len(X_train))
print("Test:", len(X_test))

# =========================
# MODEL
# =========================

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=(
        len(y_train[y_train == 0]) /
        len(y_train[y_train == 1])
    ),
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(
    "Accuracy:",
    accuracy_score(y_test, y_pred)
)

print(
    classification_report(
        y_test,
        y_pred
    )
)

# =========================
# SHAP
# =========================

test_df = df[df["participant_id"].isin(test_ids)].copy()
test_df = test_df.loc[X_test.index]

X_test_fixed = X_test.copy()
X_test_fixed["gender"] = test_df["gender"]

X_test_fixed = X_test_fixed[X_train.columns]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_fixed)

drop_cols = [
    col for col in X_test_fixed.columns
    if "agreement" in col or col == "gender"
]

X_plot = X_test_fixed.drop(columns=drop_cols)

keep_idx = [
    i for i, col in enumerate(X_test_fixed.columns)
    if col not in drop_cols
]

shap_plot = shap_values[:, keep_idx]

male_idx = X_test_fixed["gender"] == 1
female_idx = X_test_fixed["gender"] == 0

# =========================
# OVERALL BAR
# =========================

plt.figure()
shap.summary_plot(
    shap_plot,
    X_plot,
    plot_type="bar",
    show=False
)
plt.title("Overall Feature Contribution")
plt.show()

# =========================
# MALE BAR
# =========================

plt.figure()
shap.summary_plot(
    shap_plot[male_idx],
    X_plot[male_idx],
    plot_type="bar",
    show=False
)
plt.title("Male Feature Contribution")
plt.show()

# =========================
# FEMALE BAR
# =========================

plt.figure()
shap.summary_plot(
    shap_plot[female_idx],
    X_plot[female_idx],
    plot_type="bar",
    show=False
)
plt.title("Female Feature Contribution")
plt.show()

# =========================
# BEESWARM PLOTS
# =========================

plt.figure()
shap.summary_plot(shap_plot, X_plot)
plt.title("Overall Feature Impact")
plt.show()

plt.figure()
shap.summary_plot(
    shap_plot[male_idx],
    X_plot[male_idx]
)
plt.title("Male Feature Impact")
plt.show()

plt.figure()
shap.summary_plot(
    shap_plot[female_idx],
    X_plot[female_idx]
)
plt.title("Female Feature Impact")
plt.show()
