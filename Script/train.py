"""
===============================================================================
TRAINING SCRIPT
Vehicle Damage Classification – Insurance Claims
===============================================================================

Responsibilities:
- Load TRAIN and VALIDATION data only
- Build ResNet50 transfer learning model
- Sequential hyperparameter tuning (LR → Dropout)
- Save best models with traceable filenames
- Generate plots for audit justification

IMPORTANT:
- Test data is NOT used here
- No inference / Flask / prediction logic included
===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import image_dataset_from_directory

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =============================================================================
# CONFIGURATION (CENTRALIZED)
# =============================================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 3  # no_damage, minor_damage, major_damage

DATA_DIR = "../Data/claims_data"
MODEL_DIR = "../Notebook"
os.makedirs(MODEL_DIR, exist_ok=True)

# Hyperparameters
LEARNING_RATES = [1e-2, 1e-3, 1e-4]
FIXED_DROPOUT = 0.5
DROPOUT_RATES = [0.3, 0.5, 0.6]

# =============================================================================
# DATA LOADING
# =============================================================================
def load_datasets():
    train_ds = image_dataset_from_directory(
        os.path.join(DATA_DIR, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    val_ds = image_dataset_from_directory(
        os.path.join(DATA_DIR, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=SEED
    )

    return train_ds, val_ds

# =============================================================================
# DATA AUGMENTATION (TRAIN ONLY)
# =============================================================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# =============================================================================
# MODEL BUILDER
# =============================================================================
def build_model(learning_rate, dropout_rate):
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=IMG_SIZE + (3,)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =============================================================================
# MAIN TRAINING LOGIC
# =============================================================================
def main():
    train_ds, val_ds = load_datasets()

    # -------------------------------------------------------------------------
    # PHASE 1: LEARNING RATE TUNING
    # -------------------------------------------------------------------------
    lr_histories = []
    best_lr = None
    best_val_acc = 0.0

    for lr in LEARNING_RATES:
        print(f"\n=== Phase 1 | Training with LR = {lr} ===")

        model = build_model(learning_rate=lr, dropout_rate=FIXED_DROPOUT)

        model_path = os.path.join(
            MODEL_DIR, f"best_model_lr_{lr}.keras"
        )

        callbacks = [
            ModelCheckpoint(
                model_path,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )

        peak_acc = max(history.history["val_accuracy"])
        lr_histories.append((lr, history))

        if peak_acc > best_val_acc:
            best_val_acc = peak_acc
            best_lr = lr

    print(f"\n✅ Best Learning Rate Selected: {best_lr}")

    # Plot LR comparison
    plt.figure(figsize=(8, 5))
    for lr, history in lr_histories:
        plt.plot(history.history["val_accuracy"], label=f"LR={lr}")
    plt.title("Validation Accuracy – Learning Rate Tuning")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()

    # -------------------------------------------------------------------------
    # PHASE 2: DROPOUT TUNING
    # -------------------------------------------------------------------------
    dr_histories = []
    best_dropout = None
    best_val_acc = 0.0

    for dr in DROPOUT_RATES:
        print(f"\n=== Phase 2 | Training with Dropout = {dr} ===")

        model = build_model(learning_rate=best_lr, dropout_rate=dr)

        model_path = os.path.join(
            MODEL_DIR, f"best_model_lr_{best_lr}_dropout_{dr}.keras"
        )

        callbacks = [
            ModelCheckpoint(
                model_path,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )

        peak_acc = max(history.history["val_accuracy"])
        dr_histories.append((dr, history))

        if peak_acc > best_val_acc:
            best_val_acc = peak_acc
            best_dropout = dr

    print(f"\n✅ Best Dropout Selected: {best_dropout}")

    # Plot Dropout comparison
    plt.figure(figsize=(8, 5))
    for dr, history in dr_histories:
        plt.plot(history.history["val_accuracy"], label=f"Dropout={dr}")
    plt.title("Validation Accuracy – Dropout Tuning")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()

    print("\n🎯 Training complete.")
    print(f"Final selected model: best_model_lr_{best_lr}_dropout_{best_dropout}.keras")

# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
