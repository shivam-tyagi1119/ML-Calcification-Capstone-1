# Vehicle Damage Classification for Insurance Claims 🚗🛠️

## Project Overview

This project implements a **Deep Learning solution** for **vehicle damage classification**, designed specifically for **insurance claims processing**.  

The system analyzes uploaded vehicle images and classifies damage into **three severity levels**:

- **no_damage** – Vehicle with no visible damage  
- **minor_damage** – Dents, scratches, or small impact damage  
- **major_damage** – Significant structural or impact damage  

This enables insurance providers to:

- Automate claim triage  
- Reduce adjuster workload  
- Accelerate claim settlement  
- Maintain auditability and transparency  

The solution uses **Transfer Learning (ResNet50)**, **sequential hyperparameter tuning**, and **model checkpointing** to ensure reproducible and explainable results.

---

## Business Context & Decision Logic

| Model Prediction   | Description                           | Business Action                                  |
|------------------|---------------------------------------|-------------------------------------------------|
| `no_damage`       | Vehicle with no visible damage         | Auto-Reject / Close Claim                       |
| `minor_damage`    | Minor dents, scratches, or small impact | Route to partner repair network (Self-Service) |
| `major_damage`    | Significant structural or impact damage | Dispatch field adjuster (Escalate)             |

---

## Project Structure  
ML-CALCIFICATION-CAPSTONE-1/  
├── Data/  
│ ├── claims_data/ # Manually generated dataset (train, val, test)  
│ │ ├── train/  
│ │ ├── val/  
│ │ └── test/  
│ └── generate_sample_claims_data.py # Script to generate sample dataset  
├── Notebook/  
│ ├── notebook.  pynb # Exploratory Data Analysis (EDA)
│ ├── best_model_lr_*.keras # Model checkpoints from sequential tuning  
├── Script/  
│ ├── train.py # Training script with hyperparameter tuning  
│ ├── predict.py # Flask REST API for inference  
│ └── evaluate.py # Evaluation script  
├── Dockerfile # Docker containerization  
├── flask_ping.py # Health check endpoint for Flask  
├── README.md # Project documentation  
└── requirements.txt # Python dependencies  


> **Note:** The sample dataset in `Data/claims_data` is **manually generated** using the Python script `generate_sample_claims_data.py`.

---

## Dataset Description

### Classes

| Class Name | Description |
|-----------|-------------|
| `whole`  | Vehicle with no visible damage |
| `damage` | Dents, scratches, or small impact damage |

### Directory Structure  

claims_data/  
├── train/  
│ ├── damage/  
│ └── whole/  
├── val/  
│ ├── damage/  
│ └── whole/  
├── test/  
│ ├── damage/  
│ └── whole/  


---

## Model Architecture

- **Framework**: TensorFlow 2.15+ / Keras  
- **Base Model**: ResNet50 (pre-trained on ImageNet)  
- **Approach**: Transfer Learning  

### Components
1. Pre-trained ResNet50 backbone (initially frozen)  
2. Global Average Pooling  
3. Fully Connected Dense Layer (ReLU)  
4. Dropout Layer (regularization)  
5. Softmax Output Layer (3 classes)  

---

## Techniques Used

- **Transfer Learning**: Uses ImageNet weights to reduce training time  
- **Data Augmentation**: Random flips, rotations, zooms on training data  
- **Sequential Hyperparameter Tuning**: Tune learning rate first, then dropout  
- **Model Checkpointing**: Saves best-performing model automatically  
- **Early Stopping**: Stops training to prevent overfitting  

---

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt


2. Generate Sample Dataset
python Data/generate_sample_claims_data.py

3. Train the Model
python Script/train.py

4. Evaluate Model
python Script/evaluate.py

5. Run Prediction API
python Script/predict.py

Example cURL Request
curl -X POST http://localhost:9696/predict \
     -F "file=@path/to/vehicle_image.jpg"

Example Python Request
import requests

url = "http://localhost:9696/predict"
files = {"file": open("vehicle_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())


Sample Response:

{
  "prediction": "minor_damage",
  "confidence": 0.87
}

Docker Deployment

Build and run the container:

docker build -t vehicle-damage-api .
docker run -p 9696:9696 vehicle-damage-api


