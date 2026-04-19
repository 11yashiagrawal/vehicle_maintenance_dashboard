# 🏗️ System Architecture

This document provides the high-fidelity architecture diagram for the **Vehicle Maintenance Dashboard**. 

> [!NOTE]
> This diagram is provided as a pre-rendered image for immediate review and as a Mermaid source file for future editing.

## Architecture Diagram (Image)

![Vehicle Maintenance Architecture](./assets/architecture_diagram.png)

---

## Downloads & Source

*   **Final Image**: [architecture_diagram.png](./assets/architecture_diagram.png)
*   **Mermaid Source**: [architecture_diagram.mmd](./assets/architecture_diagram.mmd)

---

## Git Compatibility & Rendering

To ensure the diagram is always visible when pushed to Git:
1.  **GitHub/GitLab**: The Mermaid block below will render dynamically.
2.  **Fallback**: If dynamic rendering is unavailable, the image above serves as the "Production-Ready" visualization.

### Mermaid Source (Dynamic Render)

```mermaid
graph TD
    %% Styling
    classDef ui fill:#f0f9ff,stroke:#0369a1,stroke-width:2px;
    classDef logic fill:#f5f3ff,stroke:#6d28d9,stroke-width:2px;
    classDef ml fill:#fff7ed,stroke:#c2410c,stroke-width:2px;
    classDef data fill:#f0fdf4,stroke:#15803d,stroke-width:2px;
    classDef ext fill:#fafafa,stroke:#262626,stroke-width:2px,stroke-dasharray: 5 5;

    subgraph Presentation_Layer [Presentation Layer - Streamlit]
        UI_Home["Home.py"]:::ui
        UI_Pred["prediction.py"]:::ui
        UI_Agent["4_Agent_Assistant.py"]:::ui
    end

    subgraph Agent_Orchestration [Agentic Orchestration - LangGraph]
        direction TB
        LG["StateGraph<br/>(langgraph_agent.py)"]:::logic
        N1["Validate Node"]:::logic
        N2["Score Node"]:::logic
        N3["Retrieve Node"]:::logic
        N4["Report Node"]:::logic
        
        LG --> N1 --> N2 --> N3 --> N4 --> LG
    end

    subgraph Inference_Layer [Inference & Processing Layer]
        PP["preprocessor.py"]:::ml
        MT["model_tool.py"]:::ml
        XG{{"XGBoost Model<br/>(vehicle_model.joblib)"}}:::ml
        EN{{"Label Encoder<br/>(vehicle_encoder.joblib)"}}:::ml
    end

    subgraph Knowledge_Layer [Knowledge & RAG Layer]
        RT["retriever.py"]:::data
        FAISS[("FAISS / Keyword Index")]:::data
        DOCS["maintenance_docs.txt"]:::data
    end

    %% Interactions
    UI_Agent -- "User Input" --> LG
    N1 -- "Normalize & Parse" --> PP
    N2 -- "Predict Risk" --> MT
    MT -- "Inference" --> XG
    MT -- "Encoding" --> EN
    N3 -- "Fetch Context" --> RT
    RT -- "Search" --> FAISS
    FAISS -- "Source Data" --> DOCS
    LG -- "JSON Analysis" --> UI_Agent

    %% Helper Relationship
    MT -- "Feature Eng" --> PP
```

## 🏗️ Layer Descriptions

*   **Presentation Layer**: Streamlit entry points and interactive dashboards.
*   **Agentic Orchestration**: LangGraph state machine directing the diagnostic workflow.
*   **Inference Layer**: XGBoost model and custom feature engineering logic.
*   **Knowledge Layer**: RAG-based context retrieval from vehicle manuals.
