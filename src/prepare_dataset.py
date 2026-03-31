import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
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

WINDOW_SEC = 5
STRIDE_SEC = 1

EMPTY_KEEP_PROB = 0.3  # keep 30% empty windows

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("processed_data", exist_ok=True)

# ==============================
# LOAD CSV
# ==============================
df = pd.read_csv(CSV_PATH)

df["start_sec"] = pd.to_timedelta(df["start"]).dt.total_seconds()
df["end_sec"] = pd.to_timedelta(df["end"]).dt.total_seconds()

print(f"📊 Total labeled segments: {len(df)}")

# ==============================
# LABEL ENCODING
# ==============================
label_list = sorted(df["primary_label"].unique())
label_map = {label: i for i, label in enumerate(label_list)}

print(f"🐦 Total classes: {len(label_list)}")

# ==============================
# RESUME SUPPORT
# ==============================
def get_existing_state():
    files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".npy")]
    if not files:
        return 0

    indices = []
    for f in files:
        try:
            indices.append(int(f.split(".")[0]))
        except:
            continue

    return max(indices) + 1 if indices else 0


# ==============================
# WINDOW GENERATION
# ==============================
def generate_windows(audio, sr):
    total_sec = len(audio) / sr
    for start in np.arange(0, total_sec - WINDOW_SEC, STRIDE_SEC):
        yield start, start + WINDOW_SEC


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

    if mel_db.shape[1] < TARGET_TIME:
        pad = TARGET_TIME - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad), (0, 0)))
    else:
        mel_db = mel_db[:, :TARGET_TIME, :]

    return mel_db.astype(np.float32)


# ==============================
# LABEL GENERATION
# ==============================
def get_window_label(group, start_sec, end_sec):
    labels = np.zeros(len(label_map))

    if len(group) == 0:
        return labels

    for _, row in group.iterrows():
        if not (end_sec < row["start_sec"] or start_sec > row["end_sec"]):
            labels[label_map[row["primary_label"]]] = 1

    return labels


# ==============================
# MAIN PROCESSING
# ==============================
def process_data():
    metadata = []
    file_counter = get_existing_state()

    all_files = os.listdir(BASE_AUDIO_PATH)

    for filename in tqdm(all_files):

        if file_counter >= MAX_SAMPLES:
            break

        try:
            filepath = os.path.join(BASE_AUDIO_PATH, filename)
            audio, sr = librosa.load(filepath, sr=32000)

            group = df[df["filename"] == filename]

            # optional augmentation
            if np.random.rand() < 0.3:
                audio = librosa.effects.time_stretch(audio, rate=1.1)

            for start_sec, end_sec in generate_windows(audio, sr):

                if file_counter >= MAX_SAMPLES:
                    break

                spec = create_spectrogram(audio, sr, start_sec, end_sec)
                if spec is None:
                    continue

                labels = get_window_label(group, start_sec, end_sec)

                # handle empty windows
                if labels.sum() == 0:
                    if np.random.rand() > EMPTY_KEEP_PROB:
                        continue

                file_id = f"{file_counter}.npy"
                save_path = os.path.join(SAVE_DIR, file_id)

                np.save(save_path, spec, allow_pickle=False)

                metadata.append({
                    "file": file_id,
                    "labels": labels.tolist(),
                    "filename": filename
                })

                file_counter += 1

        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

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
# SAVE NUMPY ARRAYS
# ==============================
def save_numpy(df, prefix):
    paths = df["file"].apply(
        lambda x: os.path.join(SAVE_DIR, x)
    ).values

    labels = np.stack(df["labels"].values)

    np.save(f"processed_data/{prefix}_paths.npy", paths)
    np.save(f"processed_data/{prefix}_labels.npy", labels)

    print(f"✅ Saved {prefix}: {len(paths)} samples")


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    print("🚀 Starting dataset generation...")

    new_metadata = process_data()

    if len(new_metadata) == 0:
        raise ValueError("❌ No samples generated — check paths or CSV")

    # safe metadata load
    if os.path.exists(METADATA_PATH) and os.path.getsize(METADATA_PATH) > 0:
        old = pd.read_csv(METADATA_PATH)
        final_metadata = pd.concat([old, new_metadata], ignore_index=True)
    else:
        final_metadata = new_metadata

    final_metadata.to_csv(METADATA_PATH, index=False)

    print(f"📦 Total samples: {len(final_metadata)}")

    print("🔀 Splitting dataset...")
    train_df, val_df = split_dataset(final_metadata)

    print("💾 Saving datasets...")
    save_numpy(train_df, "train")
    save_numpy(val_df, "val")

    print("🎉 DONE — Dataset ready!")