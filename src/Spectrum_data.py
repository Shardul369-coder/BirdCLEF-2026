import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import ast

# ==============================
# CONFIG
# ==============================
BASE_AUDIO_PATH = "birdclef-2026/train_soundscapes/"
CSV_PATH = "birdclef-2026/train_soundscapes_labels.csv"

SAVE_DIR = "processed_data/spectrograms"
METADATA_PATH = "processed_data/metadata.csv"

MAX_SAMPLES = 100000

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("processed_data", exist_ok=True)

# ==============================
# LOAD CSV
# ==============================
df = pd.read_csv(CSV_PATH)

df["start_sec"] = pd.to_timedelta(df["start"]).dt.total_seconds()
df["end_sec"] = pd.to_timedelta(df["end"]).dt.total_seconds()

# ==============================
# RESUME
# ==============================
def get_existing_state():
    existing_files = os.listdir(SAVE_DIR)

    if len(existing_files) == 0:
        return 0

    indices = [int(f.split(".")[0]) for f in existing_files if f.endswith(".npy")]
    max_index = max(indices)

    print(f"🔁 Resuming from {max_index + 1}")
    return max_index + 1

# ==============================
# SPECTROGRAM
# ==============================
def create_spectrogram(audio, sr, start_sec, end_sec):
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)

    segment = audio[start_sample:end_sample]

    if len(segment) == 0:
        return None

    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=sr,
        n_fft=1024,
        hop_length=320,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    return np.expand_dims(mel_db, axis=-1).astype(np.float32)

# ==============================
# PROCESS (CSV-DRIVEN)
# ==============================
def process_labeled():
    metadata = []

    file_counter = get_existing_state()

    grouped = df.groupby("filename")

    for filename, group in tqdm(grouped):
        if file_counter >= MAX_SAMPLES:
            break

        try:
            filepath = os.path.join(BASE_AUDIO_PATH, filename)
            audio, sr = librosa.load(filepath, sr=32000)

            for _, row in group.iterrows():
                if file_counter >= MAX_SAMPLES:
                    break

                spec = create_spectrogram(
                    audio,
                    sr,
                    row["start_sec"],
                    row["end_sec"]
                )

                if spec is None:
                    continue

                file_id = f"{file_counter}.npy"
                save_path = os.path.join(SAVE_DIR, file_id)

                np.save(save_path, spec, allow_pickle=False)

                # ✅ FIX: add labels
                label = ast.literal_eval(row["label_vector"])

                metadata.append({
                    "file": file_id,
                    "labels": label
                })

                file_counter += 1

        except Exception as e:
            print(f"Error: {filename} -> {e}")

    return pd.DataFrame(metadata)

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    new_metadata = process_labeled()

    if os.path.exists(METADATA_PATH):
        old = pd.read_csv(METADATA_PATH)
        final = pd.concat([old, new_metadata], ignore_index=True)
    else:
        final = new_metadata

    final.to_csv(METADATA_PATH, index=False)

    print(f"✅ Total samples: {len(final)}")