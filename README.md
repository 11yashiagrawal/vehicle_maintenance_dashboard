🚗 Vehicle Maintenance Prediction
& Agentic Fleet Management System
An end-to-end predictive maintenance system that proactively detects vehicles requiring servicing using structured telemetry data.

Built with progressive model comparison, hyperparameter tuning, and deployed using Streamlit for real-time inference.

🌐 Live System

🔗 Streamlit App: [Add your deployed link]
🎥 Project Demo Video: [Add your video link]

🎯 Problem Statement

Fleet management systems traditionally rely on:

Fixed servicing schedules

Reactive maintenance

Manual inspection

These approaches ignore real operational stress.

This project answers:

Can we use telemetry data to predict maintenance requirements before failure occurs?

This is not just classification — it is risk detection.

False negatives may lead to:

Breakdown

Downtime

Financial loss

Safety hazards

Therefore, model selection prioritizes Recall and F1-score, not accuracy alone.

📊 Dataset Overview
Metric	Value
Samples	~15,000 vehicles
Target	maintenance_required
Imbalance	75% : 25%
Final Features	28
Feature Types	Numerical + One-Hot Encoded
🧠 Feature Engineering Strategy

Raw telemetry cannot fully capture mechanical stress.

Engineered features include:

Feature	Captures
mileage_per_year	Operational intensity
thermal_stress	Load × Temperature strain
engine_hours_per_km	Efficiency workload
fault_density	Fault recurrence
load_efficiency	Stress relative to output

Feature importance confirmed these transformations.

🤖 Model Development Pipeline

Progressive modelling approach:

Logistic Regression (baseline linear)

Decision Tree (non-linear splits)

Random Forest (variance reduction)

XGBoost (final tuned model)

📈 Model Performance (Test Set)
Model	Accuracy	Precision	Recall	F1	ROC-AUC
Logistic Regression	0.84	0.76	0.60	0.68	0.92
Decision Tree	0.83	0.73	0.60	0.66	0.85
Random Forest	0.85	0.78	0.58	0.70	0.91
XGBoost (Tuned)	0.85	0.80	0.76	0.78	0.92
Why XGBoost?

Highest recall (critical for maintenance detection)

Balanced precision-recall

Strong ROC-AUC

Better minority class detection

Selected based on operational risk considerations.

⚙️ Hyperparameter Optimization

RandomizedSearchCV tuned:

max_depth

n_estimators

learning_rate

subsample

scale_pos_weight

scale_pos_weight handled class imbalance.

🏗 System Architecture
Raw Data
   ↓
EDA & Cleaning
   ↓
Feature Engineering
   ↓
Encoding
   ↓
Train-Test Split
   ↓
Model Training (4 Models)
   ↓
Hyperparameter Tuning
   ↓
Evaluation
   ↓
Model Serialization
   ↓
Streamlit Deployment
🚀 Deployment Architecture
🔐 Model Serialization

Saved using joblib

Ensures reproducibility

📦 Feature Order Preservation

features.csv exported

Prevents inference misalignment

🖥 Streamlit UI

Numeric telemetry inputs

Dropdown categorical selection

Real-time risk prediction

Probability output

📊 Probability-Based Decision

Instead of returning 0/1 only:

P(Maintenance Required)

Allows dynamic threshold tuning.

📂 Repository Structure
Vehicle-Maintenance-Prediction/
│
├── notebooks/          # Colab experimentation & EDA
├── models/             # Trained model files (.joblib)
├── data/               # Processed dataset
├── app.py              # Streamlit deployment
├── features.csv        # Feature order reference
├── requirements.txt    # Dependencies
├── report/             # LaTeX + PDF report
└── README.md
🧰 Technology Stack
Layer	Tools
Data Processing	pandas, NumPy
Visualization	matplotlib, seaborn
Machine Learning	scikit-learn, XGBoost
Tuning	RandomizedSearchCV
Deployment	Streamlit
Hosting	Streamlit Cloud
Version Control	Git, GitHub