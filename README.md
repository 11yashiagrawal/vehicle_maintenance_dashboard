# 🚗 Vehicle Maintenance Prediction & Agentic Fleet Management

An end-to-end machine learning system that predicts whether a vehicle requires maintenance using structured operational telemetry data.  
The pipeline performs exploratory data analysis, feature engineering, multi-model comparison, hyperparameter tuning, and deploys a tuned XGBoost classifier through an interactive Streamlit application.

> **Live Demo:** [[Deployed Web App]](https://vehicle-maintenance-dashboard.streamlit.app/)  
> **Project Video:** [[Project Demo Link]](https://drive.google.com/file/d/1Zo7nhaxWIxzFFTYz8joRYZw6oG889lui/view?usp=drive_link)
> **Colab Notebook Link:** [[Google Colab Notebook]](https://colab.research.google.com/drive/1GCYD7v7glAUhZDgZqOBbnpfGuIdSKBGr?usp=sharing)

---

## Highlights

|                        |                                                            |
| ---------------------- | ---------------------------------------------------------- |
| **Problem Type**       | Predictive Maintenance (Binary Classification)             |
| **Dataset Size**       | ~15,000 records, 28 engineered features                    |
| **Class Distribution** | ~75% No Maintenance / ~25% Maintenance                     |
| **Models Compared**    | Logistic Regression, Decision Tree, Random Forest, XGBoost |
| **Final Model**        | Tuned XGBoost (ROC-AUC = 0.92)                             |
| **Stack**              | Python, pandas, scikit-learn, XGBoost, Streamlit           |
| **Deployment**         | Streamlit Cloud                                            |

---

## Problem Statement

Fleet management systems traditionally rely on fixed schedules and reactive servicing.  
These approaches fail to consider real operational stress and vehicle-specific usage patterns.

This project addresses the following core question:

> Can telemetry data be used to proactively detect vehicles requiring maintenance before breakdown occurs?

This is not simply a classification task — it is a **risk detection problem**, where false negatives may lead to:

- Breakdown
- Operational downtime
- Financial loss
- Safety hazards

Therefore, model evaluation prioritizes **Recall and F1-score**, not accuracy alone.

---

## Dataset Overview

The dataset consists of structured fleet telemetry features including:

- mileage_km
- engine_hours
- vehicle_age_years
- fault_code_count
- oil_temp_avg_celsius
- vibration_level
- battery_voltage
- engine_load_percent
- fuel_efficiency_kmpl
- days_since_last_service

Additional domain-driven engineered features were created to capture operational intensity and mechanical stress interactions.

---

## Feature Engineering

Raw telemetry alone does not capture compounded stress effects.  
The following engineered features were introduced:

| Engineered Feature        | Purpose                        |
| ------------------------- | ------------------------------ |
| mileage_per_year          | Captures usage intensity       |
| thermal_stress            | Engine load × oil temperature  |
| engine_hours_per_km       | Efficiency indicator           |
| fault_density             | Normalized fault frequency     |
| load_efficiency           | Stress relative to performance |
| days_since_last_service   | Time since last service        |

Feature importance analysis confirmed the predictive value of these transformations.

---

## Model Development Strategy

A progressive modelling approach was followed:

1. Logistic Regression — linear baseline
2. Decision Tree — non-linear baseline
3. Random Forest — ensemble variance reduction
4. **XGBoost — tuned final model**

---

## Model Performance (Test Set)

| Model               | Accuracy | Precision | Recall   | F1       | ROC-AUC  |
| ------------------- | -------- | --------- | -------- | -------- | -------- |
| Logistic Regression | 0.84     | 0.76      | 0.60     | 0.68     | 0.92     |
| Decision Tree       | 0.83     | 0.73      | 0.60     | 0.66     | 0.85     |
| Random Forest       | 0.85     | 0.78      | 0.58     | 0.70     | 0.91     |
| **XGBoost (Tuned)** | **0.85** | **0.80**  | **0.76** | **0.78** | **0.92** |

XGBoost was selected due to its superior recall and balanced performance across evaluation metrics.

---

## Hyperparameter Optimization

RandomizedSearchCV was used to tune:

- max_depth
- n_estimators
- learning_rate
- subsample
- scale_pos_weight

Class imbalance was handled through optimized `scale_pos_weight`.

---

## Deployment

The final tuned XGBoost model was serialized using `joblib` and deployed via Streamlit.

Deployment ensures:

- Feature order preservation
- Probability-based risk scoring
- Real-time inference

The deployed application allows fleet operators to input telemetry data and receive maintenance risk predictions instantly.

---

## 🚀 Getting Started

Follow these steps to set up and run the application on your local machine.

### 1. Fork the Repository

- Click the **Fork** button at the top-right of this page.
- Select your GitHub account to create a copy of this repository.

### 2. Clone the Repository

Open your terminal and run the following command (replace `[your-username]` with your actual GitHub username):

```bash
git clone https://github.com/[your-username]/vehicle_maintenance_dashboard.git
cd vehicle_maintenance_dashboard
```

### 3. Set Up a Virtual Environment (Recommended)

It's best practice to use a virtual environment to avoid dependency conflicts:

```bash
# Create the environment
python -m venv venv

# Activate it (Mac/Linux)
source venv/bin/activate

# Activate it (Windows)
# venv\Scripts\activate
```

### 4. Install Dependencies

Install all the required Python libraries:

```bash
pip install -r requirements.txt
```

### 5. Run the Application

You can launch the application using:

```bash
streamlit run Home.py
```

---

## 📂 Repository Structure

```
.
├── Home.py               # Main landing page for the portal
├── app.py                # Monolithic fallback prediction app
├── requirements.txt      # Python dependencies
├── assets/               # CSS styles and background images
├── data/                 # Raw/Sample datasets
├── models/               # Serialized XGBoost model artifacts
├── pages/                # Streamlit sub-pages (EDA, Prediction, Insights)
└── utils/                # Shared logic (preprocessor, model loader, ui styling)
└── .streamlit/
    └── config.toml/      # Configuration of global colors and font      
```

---

## 🛠️ Usage

1. **Home**: High-level overview of the system.
2. **Data Insights**: Interactive EDA including class balance and dynamic boxplots for outlier detection.
3. **Model Insights**: Detailed breakdown of the engineered features and model strategy.
4. **Prediction**: Live risk-scoring interface. Select your vehicle details and last service date to receive a pro-active maintenance recommendation.
