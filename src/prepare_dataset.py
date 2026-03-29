import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import ast
from sklearn.model_selection import train_test_split

# ==============================
# CONFIG
# ==============================
BASE_AUDIO_PATH = "birdclef-2026/train_soundscapes/"
CSV_PATH = "birdclef-2026/train_soundscapes_labels.csv"

SAVE_DIR = "processed_data/spectrograms"
METADATA_PATH = "processed_data/metadata.csv"

MAX_SAMPLES = 100000
TARGET_TIME = 500

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("processed_data", exist_ok=True)

# ==============================
# LOAD CSV
# ==============================
df = pd.read_csv(CSV_PATH)

df["start_sec"] = pd.to_timedelta(df["start"]).dt.total_seconds()
df["end_sec"] = pd.to_timedelta(df["end"]).dt.total_seconds()

# 🔥 Shuffle to avoid bias
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ==============================
# RESUME
# ==============================
def get_existing_state():
    existing_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".npy")]

    if not existing_files:
        return 0

    indices = []
    for f in existing_files:
        try:
            indices.append(int(f.split(".")[0]))
        except:
            continue

    if not indices:
        return 0

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

    mel_db = np.expand_dims(mel_db, axis=-1)

    # FIX SHAPE
    if mel_db.shape[1] < TARGET_TIME:
        pad = TARGET_TIME - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0),(0,pad),(0,0)))
    else:
        mel_db = mel_db[:, :TARGET_TIME, :]

    return mel_db.astype(np.float32)

# ==============================
# CREATE SPECTROGRAM DATA
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

                label = ast.literal_eval(row["label_vector"])

                metadata.append({
                    "file": file_id,
                    "labels": label,
                    "filename": filename   # 🔥 IMPORTANT
                })

                file_counter += 1

        except Exception as e:
            print(f"⚠️ Error: {filename} -> {e}")

    return pd.DataFrame(metadata)

# ==============================
# SPLIT DATASET
# ==============================
def split_dataset(metadata_df):

    unique_files = metadata_df["filename"].unique()

    train_files, val_files = train_test_split(
        unique_files,
        test_size=0.2,
        random_state=42
    )

    train_df = metadata_df[metadata_df["filename"].isin(train_files)]
    val_df = metadata_df[metadata_df["filename"].isin(val_files)]

    return train_df, val_df

# ==============================
# SAVE NUMPY FILES
# ==============================
def save_numpy(df, prefix):

    paths = df["file"].apply(
        lambda x: os.path.join(SAVE_DIR, x)
    ).values

    labels = np.stack(
        df["labels"].apply(ast.literal_eval).values
    )

    np.save(f"processed_data/{prefix}_paths.npy", paths)
    np.save(f"processed_data/{prefix}_labels.npy", labels)

    print(f"✅ Saved {prefix}: {len(paths)} samples")

# ==============================
# MAIN PIPELINE
# ==============================
if __name__ == "__main__":

    print("🚀 Starting preprocessing...")

    new_metadata = process_labeled()

    if os.path.exists(METADATA_PATH):
        old = pd.read_csv(METADATA_PATH)
        final_metadata = pd.concat([old, new_metadata], ignore_index=True)
    else:
        final_metadata = new_metadata

    final_metadata.to_csv(METADATA_PATH, index=False)

    print(f"✅ Total samples: {len(final_metadata)}")

    print("🔀 Splitting dataset...")
    train_df, val_df = split_dataset(final_metadata)

    print("💾 Saving datasets...")
    save_numpy(train_df, "train")
    save_numpy(val_df, "val")

    print("🎉 Data pipeline complete!")