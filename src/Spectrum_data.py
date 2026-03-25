import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
BASE_AUDIO_PATH = "birdclef-2026/train_soundscapes/"
SAVE_DIR = "processed_data/spectrograms"
METADATA_PATH = "processed_data/metadata.csv"

SEGMENT_DURATION = 5
MAX_SAMPLES = 100000  # change later → resume automatically

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("processed_data", exist_ok=True)

# ==============================
# LOAD EXISTING STATE
# ==============================
def get_existing_state():
    existing_files = os.listdir(SAVE_DIR)

    if len(existing_files) == 0:
        return 0, set()

    indices = [int(f.split(".")[0]) for f in existing_files if f.endswith(".npy")]
    max_index = max(indices)

    print(f"🔁 Resuming from sample: {max_index + 1}")

    return max_index + 1, set(indices)


# ==============================
# SPECTROGRAM FUNCTION
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
# MAIN PROCESSING
# ==============================
def process_full_audio():
    metadata = []

    # 🔁 Resume support
    file_counter, existing_indices = get_existing_state()

    files = sorted(os.listdir(BASE_AUDIO_PATH))

    print(f"🚀 Processing {len(files)} audio files...")

    for filename in tqdm(files):
        if file_counter >= MAX_SAMPLES:
            break

        try:
            filepath = os.path.join(BASE_AUDIO_PATH, filename)
            audio, sr = librosa.load(filepath, sr=32000)

            total_duration = len(audio) / sr
            num_segments = int(total_duration // SEGMENT_DURATION)

            for i in range(num_segments):
                if file_counter >= MAX_SAMPLES:
                    break

                # Skip already processed index
                if file_counter in existing_indices:
                    file_counter += 1
                    continue

                start = i * SEGMENT_DURATION
                end = start + SEGMENT_DURATION

                spec = create_spectrogram(audio, sr, start, end)

                if spec is None:
                    continue

                file_id = f"{file_counter}.npy"
                save_path = os.path.join(SAVE_DIR, file_id)

                np.save(save_path, spec, allow_pickle=False)

                metadata.append({
                    "file": file_id,
                    "filename": filename,
                    "start": start,
                    "end": end
                })

                file_counter += 1

        except Exception as e:
            print(f"Error: {filename} -> {e}")

    return pd.DataFrame(metadata)


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    new_metadata = process_full_audio()

    # 🔁 Append metadata instead of overwrite
    if os.path.exists(METADATA_PATH):
        old_metadata = pd.read_csv(METADATA_PATH)
        final_metadata = pd.concat([old_metadata, new_metadata], ignore_index=True)
    else:
        final_metadata = new_metadata

    final_metadata.to_csv(METADATA_PATH, index=False)

    print(f"✅ Total samples now: {len(final_metadata)}")