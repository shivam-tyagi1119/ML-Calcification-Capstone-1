Vehicle Damage Classification for Insurance Claims📋 Project OverviewThis project implements a Deep Learning solution for automated vehicle damage classification, specifically designed for integration into insurance claims processing pipelines.By analyzing uploaded vehicle images, the system classifies damage into three severity levels. This enables insurance providers to:Automate Claim Triage: Instantly route claims based on predicted severity.Reduce Adjuster Workload: Filter out "no damage" or "minor" claims for automated processing.Maintain Auditability: Provide a transparent, reproducible trail of how model decisions are made.🏗 Project StructureThe repository is organized to separate data generation, model experimentation, and production-ready scripts.PlaintextML-CALCIFICATION-CAPSTONE-1/
├── Data/
│   ├── claims_data/          # Generated dataset (train, val, test)
│   └── generate_sample_claims_data.py  # Synthetic data generation script
├── Notebook/
│   ├── notebook.ipynb        # Exploratory Data Analysis (EDA)
│   ├── best_model_*.keras    # Audited model checkpoints
│   └── test_confusion_matrix.png # Evaluation visualization
├── Script/
│   ├── train.py              # Sequential hyperparameter tuning & training
│   ├── predict.py            # Flask REST API for model inference
│   └── evaluate.py           # Hold-out test set evaluation
├── Dockerfile                # Containerization for deployment
└── requirements.txt          # Project dependencies
🏛 Business Context & Decision LogicThe model outputs one of three classes, each mapped to a specific business action within the insurance workflow:Model OutputDescriptionBusiness DecisionNo DamageNo visible structural issues.Auto-Reject/Close: Fast-track for automated closure.Minor DamageDents, scratches, or light impact.Self-Service: Route to partner repair network.Major DamageSignificant structural impact.Escalate: Dispatch field adjuster for inspection.🛠 Technical SpecificationsFramework: TensorFlow 2.15+ / KerasEnvironment: Optimized for Python 3.12Architecture: ResNet50 backbone (Transfer Learning)Tuning Strategy: Sequential Tuning (Phase 1: Learning Rate → Phase 2: Dropout) to ensure audit traceability and GPU efficiency.🚀 Getting Started1. InstallationEnsure you have Python 3.12 installed. Install the necessary packages:Bashpip install -r requirements.txt
2. Dataset GenerationBefore training, generate the synthetic proxy dataset to populate the Data/claims_data directory:Bashpython Data/generate_sample_claims_data.py
3. Training the ModelRun the training script to execute the sequential tuning phases. This will save the best-performing models in the Notebook/ folder:Bashpython Script/train.py
4. Model EvaluationEvaluate the final audited model against the hold-out test set to generate accuracy reports and a confusion matrix:Bashpython Script/evaluate.py
5. Deployment (Inference)Launch the Flask-based prediction service on port 9696:Bashpython Script/predict.py
🔌 API DocumentationEndpoint: POST /predictUsed to submit a vehicle image for damage triage.Sample Request (cURL):Bashcurl -X POST http://localhost:9696/predict \
     -F "file=@path/to/your/vehicle_image.jpg"
Sample JSON Response:JSON