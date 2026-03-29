import tensorflow as tf
import numpy as np

# ==============================
# LOAD FUNCTION
# ==============================
def load_npy(path, label):
    def _load(path):
        spec = np.load(path.decode())

        if spec.shape[1] < 500:
            pad_width = 500 - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad_width), (0, 0)))
        else:
            spec = spec[:, :500, :]

        return spec.astype(np.float32)

    spec = tf.numpy_function(_load, [path], tf.float32)
    spec.set_shape((128, 500, 1))

    return spec, label


# ==============================
# DATASET BUILDER
# ==============================
def build_dataset(paths, labels, batch_size, training=True):

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(1000)

    dataset = dataset.map(load_npy, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)   # ✅ dynamic batch size
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ==============================
# MAIN FUNCTION
# ==============================
def get_dataset(batch_size=16):

    train_paths = np.load("processed_data/train_paths.npy", allow_pickle=True)
    train_labels = np.load("processed_data/train_labels.npy")

    val_paths = np.load("processed_data/val_paths.npy", allow_pickle=True)
    val_labels = np.load("processed_data/val_labels.npy")

    train_dataset = build_dataset(
        train_paths,
        train_labels,
        batch_size=batch_size,
        training=True
    )

    val_dataset = build_dataset(
        val_paths,
        val_labels,
        batch_size=batch_size,
        training=False
    )

    return train_dataset, val_dataset