# 🚗 Vehicle Maintenance Prediction & Agentic Fleet Management

An end-to-end machine learning system that predicts whether a vehicle requires maintenance using structured operational telemetry data.  
The system integrates an advanced **LangGraph-powered AI Agent** with a highly-tuned XGBoost classifier. The pipeline performs exploratory data analysis, comprehensive feature engineering, RAG-based document retrieval, and contextual maintenance triage through an interactive Streamlit application.

> **Live Demo:** [[Deployed Web App]](https://vehicle-maintenance-dashboard.streamlit.app/)  
> **Project Video:** [[Project Demo Link]](https://drive.google.com/file/d/1Zo7nhaxWIxzFFTYz8joRYZw6oG889lui/view?usp=drive_link)
> **Colab Notebook Link:** [[Google Colab Notebook]](https://colab.research.google.com/drive/1GCYD7v7glAUhZDgZqOBbnpfGuIdSKBGr?usp=sharing)
> **Project Report:** [[PDF Report]](./report/vehicle_maintenance_dashboard_report.pdf)

---

## Highlights

|                        |                                                            |
| ---------------------- | ---------------------------------------------------------- |
| **Problem Type**       | Predictive Maintenance (Binary Classification)             |
| **Dataset Size**       | ~15,000 records, 28 engineered features                    |
| **Class Distribution** | ~75% No Maintenance / ~25% Maintenance                     |
| **Models Compared**    | Logistic Regression, Decision Tree, Random Forest, XGBoost |
| **Final Model**        | Tuned XGBoost (ROC-AUC = 0.92)                             |
| **Stack**              | Python, XGBoost, Streamlit, LangGraph, LangChain, FAISS    |
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

Feature importance analysis from XGBoost confirmed that engineered interaction features such as `thermal_stress` and `fault_density` ranked among the top contributors, validating the domain-driven design approach.

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

## Why XGBoost?

XGBoost was selected because:

- It captures non-linear interactions efficiently.
- It handles feature interactions automatically.
- Built-in regularization reduces overfitting.
- scale_pos_weight improves imbalance handling.
- It achieved the highest ROC-AUC and Recall among all models.

This makes it particularly suited for predictive maintenance risk detection.
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

## 🤖 Agentic Fleet Management (LangGraph)

Beyond standard predictive modeling, this project features an embedded **LangGraph-driven AI Agent** to assist technicians and fleet operators. 

The smart maintenance agent:
- Implements a stateful Directed Acyclic Graph (DAG) for diagnostic routing.
- Uses **RAG (Retrieval-Augmented Generation)** over maintenance manuals using **FAISS** and **Sentence Transformers**.
- Dynamically generates actionable repair steps based on user queries, telemetry data, and risk predictions.
- Enforces predefined fleet safety policies through autonomous rule-based checks.

---

## System Architecture

The end-to-end pipeline follows this structured workflow:

1. Raw Fleet Telemetry Data  
2. Data Cleaning & Feature Engineering  
3. XGBoost Hyperparameter Tuning & Model Serialization
4. Local Vector Database (FAISS) Generation for Manuals
5. LangGraph Agent Pipeline Construction
6. Streamlit Deployment with Real-time ML Inference and Agent Context
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
## Technology Stack

| Layer            | Technology Used |
|------------------|-----------------|
| Data Processing  | pandas, numpy   |
| Modeling         | scikit-learn, XGBoost |
| Agentic Workflow | LangGraph, LangChain |
| RAG & Retrieval  | FAISS, SentenceTransformers |
| LLM Engine       | Ollama (Local) |
| Deployment       | Streamlit |
| Serialization    | joblib |

## 📂 Repository Structure

```text
vehicle_maintenance_dashboard/
│
├── assets/            -> Static files (CSS, images)
├── data/              -> Datasets used for training and inference
├── models/            -> Serialized trained model artifacts
├── pages/             -> Streamlit multi-page application modules
├── utils/             -> Shared preprocessing and model utilities
├── report/            -> Final project report (LaTeX + PDF)
├── walkthrough/       -> Complete ML development notebook
├── .streamlit/        -> Streamlit configuration
├── .devcontainer/     -> Development container setup
│
├── Home.py            -> Main Streamlit entry point
├── app.py             -> Alternative / fallback app interface
├── requirements.txt   -> Project dependencies
└── README.md          -> Project documentation
```
---

## 🛠️ Usage

1. **Home**: High-level overview of the system.
2. **Data Insights**: Interactive EDA including class balance and dynamic boxplots for outlier detection.
3. **Model Insights**: Detailed breakdown of the engineered features and model strategy.
4. **Prediction**: Live risk-scoring interface. Select your vehicle details and last service date to receive a pro-active maintenance recommendation.
5. **Agent Assistant**: AI-powered maintenance agent using LangGraph and RAG to provide personalized diagnostic reports, prioritize mechanical issues, and verify fleet compliance policies.
