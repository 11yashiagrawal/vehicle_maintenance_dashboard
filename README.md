# Vehicle Maintenance Prediction & Agentic Fleet Management
### Transitioning from Predictive Analytics to Autonomous Fleet Operations

## Project Vision
This system represents an end-to-end evolution of fleet management. Starting with **Milestone 1**, we establish a high-precision predictive engine using classical Machine Learning. This engine serves as the analytical foundation for **Milestone 2**, where the system will be transformed into an **Agentic AI Assistant** capable of autonomous reasoning, service scheduling, and strategic fleet optimization.

---

## Milestone 1: Predictive Analytics Engine
The objective of this milestone is to move fleet operations from **Reactive Maintenance** (fixing after failure) to **Proactive Maintenance** (predicting failure before it occurs).

### 1. Data Architecture & Signals
The model processes high-frequency telemetry data to extract signals of mechanical degradation.



#### Feature Engineering (The "Stress" Metrics)
Beyond raw data, we engineered specific domain-relevant features to capture the "Health Index" of each vehicle:
* **Thermal Stress Index:** $oil\_temp \times engine\_load$. Captures the combined impact of high temperature and high demand.
* **Fault Density:** $fault\_codes / engine\_hours$. Normalizes errors against usage time to identify chronically failing units.
* **Operational Intensity:** `mileage_per_year` and `engine_hours_per_km` to identify vehicles in extreme duty cycles.
* **Efficiency Decay:** `load_efficiency` ($load / fuel\_efficiency$) to detect abnormal fuel consumption.

### 2. Robust Preprocessing Pipeline
To ensure model reliability, a rigorous data sanitization pipeline was implemented:
* **Temporal Logic:** Converted service dates into a continuous `days_since_last_service` variable.
* **Outlier Mitigation:** Utilized **IQR Capping** for `mileage_km` and `engine_hours`.
* **Statistical Imputation:** Numerical nulls handled via **Median Imputation**; categorical nulls via **Mode Imputation**.
* **Multicollinearity Management:** Applied One-Hot Encoding with `drop_first=True`.

---

## 3. Machine Learning Strategy
### Algorithm Selection: Random Forest
While the project scope allows for Logistic Regression or Decision Trees, we implemented a **Random Forest Classifier ($n=400$)** to leverage ensemble learning.



**Why Random Forest?**
* **Non-Linear Interactions:** Mechanical failure is rarely linear; it is often the interaction of multiple stressors.
* **Feature Importance:** Allows ranking of telemetry signals (e.g., fault codes vs. oil temp) as risk predictors.
* **Class Imbalance:** Optimized using `class_weight="balanced"` to ensure critical "Maintenance Required" events are captured.

### 4. Evaluation Framework
The model is evaluated on its ability to minimize **False Negatives** (missed failures).
* **Metrics:** Accuracy, Precision, Recall, and F1-Score.
* **Splitting:** 80/20 Stratified Split to maintain class proportions.
* **Persistence:** Model and feature metadata are serialized via `joblib`.

---

## 5. Future Roadmap: Agentic Evolution (Milestone 2)
The predictive output of this model will serve as "sensory input" for an AI Agent built on **LangGraph**.

| Component | Function |
| :--- | :--- |
| **Reasoning Engine** | Open-source LLM (via Free-tier APIs) |
| **Agent Framework** | LangGraph for autonomous workflow state management |
| **Knowledge Base** | RAG (Chroma/FAISS) containing vehicle manuals and logs |
| **Output** | Autonomous service scheduling and risk explanations |

---

## Technical Stack
* **Core:** Python, Pandas, NumPy, Scikit-Learn
* **Environment:** Google Colab / GitHub
* **Deployment Target:** Streamlit / Hugging Face Spaces (Mandatory)
* **Serialization:** Joblib
