import os
import numpy as np
import pandas as pd
import librosa
import torch
from transformers import pipeline

DATASET_PATH = "audio_dataset_path"
OUTPUT_CSV = "audio_features.csv"

device = 0 if torch.cuda.is_available() else -1

clf = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er",
    device=device
)

results = []

for pid in os.listdir(DATASET_PATH):

    audio_file = os.path.join(DATASET_PATH, pid)

    if not os.path.exists(audio_file):
        continue

    y, sr = librosa.load(audio_file, sr=16000)

    y = y[:45 * sr]
    y = y / (np.max(np.abs(y)) + 1e-6)

    chunk_size = sr * 5
    chunks = [
        y[i:i + chunk_size]
        for i in range(0, len(y), chunk_size)
    ]

    vec = np.zeros(7)

    for chunk in chunks:

        if len(chunk) < sr:
            continue

        if np.mean(np.abs(chunk)) < 0.01:
            continue

        preds = clf(chunk)

        temp = np.zeros(7)

        for p in preds:

            label = p["label"].lower()
            score = p["score"]

            if "ang" in label:
                temp[0] = score

            elif "hap" in label:
                temp[3] = score

            elif "sad" in label:
                temp[5] = score

            elif "neu" in label:
                temp[4] = score

        vec += temp

    vec = vec / vec.sum()

    final = np.zeros(7)

    final[0] = vec[0]
    final[3] = vec[3]
    final[4] = vec[4]
    final[5] = vec[5]

    final[1] = 0.3 * final[0]
    final[2] = 0.3 * final[5] + 0.2 * final[0]
    final[6] = 0.3 * final[3]

    final = final / final.sum()

    results.append(final.tolist())

pd.DataFrame(results).to_csv(
    OUTPUT_CSV,
    index=False
)
