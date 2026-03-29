import os
import logging
import tensorflow as tf
from ruamel.yaml import YAML

from src.losses import get_loss
from src.dataset_loader import get_dataset

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# ==============================
# LOAD PARAMS
# ==============================
yaml_loader = YAML()
params = yaml_loader.load(open("params.yaml"))["Model"]

# ==============================
# ENV SETUP
# ==============================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ==============================
# LOGGING
# ==============================
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('Model_seg')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, 'Model_seg.log'))

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ==============================
# ATTENTION BLOCK
# ==============================
def attention_block(x):
    attn = layers.Conv2D(1, (1, 1), activation="sigmoid")(x)
    return layers.Multiply()([x, attn])

# ==============================
# BUILD MODEL
# ==============================
def build_model(input_shape=(128, 500, 1), num_classes=264):

    inputs = layers.Input(shape=input_shape)

    x = layers.Lambda(lambda x: tf.image.grayscale_to_rgb(x))(inputs)
    x = layers.Resizing(224, 224)(x)

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )

    # Freeze early layers
    for layer in base_model.layers[:200]:
        layer.trainable = False

    x = base_model.output
    x = attention_block(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    return models.Model(inputs, outputs)

# ==============================
# TRAIN MODEL
# ==============================
def train_model(model, train_dataset, val_dataset, num_classes=264):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=params["LEARNING_RATE"]
        ),
        loss=get_loss(),
        metrics=[
            tf.keras.metrics.AUC(
                curve="PR",
                multi_label=True,
                num_labels=num_classes,
                name="pr_auc"
            ),
            tf.keras.metrics.AUC(
                curve="ROC",
                multi_label=True,
                num_labels=num_classes,
                name="roc_auc"
            )
        ]
    )

    cb = [
    keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/seg_best.h5",
        save_best_only=True,
        monitor="val_pr_auc",   # ✅ FIXED
        mode="max"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        patience=3,
        factor=0.3,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",   # ✅ FIXED
        mode="max",
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
        ]

    logger.debug('Model training started')
    logger.debug(f"Training for {params['EPOCHS']} epochs...")


    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=params["EPOCHS"],
        callbacks=cb
    )


    logger.info("✅ Training completed")

    return model, history

# ==============================
# SAVE MODEL
# ==============================
def save_model(model):
    os.makedirs("checkpoints", exist_ok=True)
    path = "checkpoints/seg_best.h5"
    model.save(path)
    logger.info(f"Model saved at {path}")

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    logger.info("Loading dataset...")
    train_dataset, val_dataset = get_dataset(
        batch_size=params["BATCH_SIZE"]   # ✅ controlled here
    )

    logger.info("Building model...")
    model = build_model(num_classes=params["NUM_CLASSES"])

    logger.info("Starting training...")
    model, history = train_model(model, train_dataset, val_dataset)

    save_model(model)