"""
===============================================================================
MODEL EVALUATION SCRIPT
Vehicle Damage Classification – Insurance Claims
===============================================================================

Purpose:
- Load the final trained model
- Evaluate performance on the HOLD-OUT TEST SET
- Report unbiased test accuracy

IMPORTANT:
- Test data is used exactly once
- No training or hyperparameter tuning here
- No Flask / API logic
===============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import random
import numpy as np
import tensorflow as tf

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
# CONFIGURATION
# =============================================================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 3

DATA_DIR = "../Data/claims_data"
MODEL_PATH = "../Notebook/best_model_lr_0.01_dropout_0.3.keras"  # <-- adjust if needed

CLASS_NAMES = ["no_damage", "minor_damage", "major_damage"]

# =============================================================================
# LOAD TEST DATA
# =============================================================================
print("Loading test dataset...")

test_ds = image_dataset_from_directory(
    os.path.join(DATA_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# =============================================================================
# LOAD MODEL
# =============================================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# =============================================================================
# EVALUATE MODEL (TEST SET USED ONCE)
# =============================================================================
print("\nEvaluating model on test dataset...")
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

# =============================================================================
# REPORT RESULTS
# =============================================================================
print("\n================= TEST RESULTS =================")
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print("================================================")

print("\nEvaluation complete. Test set was used exactly once.")
