import os
import numpy as np
from PIL import Image

# -------------------------------
# CONFIGURATION
# -------------------------------
BASE_DIR = "claims_data"
SPLITS = {
    "train": 60,
    "val": 20,
    "test": 20
}
CLASSES = ["no_damage", "minor_damage", "major_damage"]
IMG_SIZE = (224, 224)
SEED = 42

np.random.seed(SEED)

# -------------------------------
# IMAGE GENERATION FUNCTION
# -------------------------------
def generate_image(class_name):
    """
    Generate a synthetic image with slight visual patterns
    to simulate different damage levels.
    """
    img = np.random.randint(0, 255, (*IMG_SIZE, 3), dtype=np.uint8)

    if class_name == "minor_damage":
        img[80:140, 80:140] = [200, 50, 50]  # red-ish patch
    elif class_name == "major_damage":
        img[50:180, 50:180] = [120, 120, 120]  # gray damaged area

    return Image.fromarray(img)

# -------------------------------
# DATASET CREATION
# -------------------------------
def create_dataset():
    for split, count in SPLITS.items():
        for cls in CLASSES:
            dir_path = os.path.join(BASE_DIR, split, cls)
            os.makedirs(dir_path, exist_ok=True)

            for i in range(count):
                img = generate_image(cls)
                img.save(os.path.join(dir_path, f"{cls}_{i}.jpg"))

    print("✅ Sample claims dataset created successfully")

# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    create_dataset()
