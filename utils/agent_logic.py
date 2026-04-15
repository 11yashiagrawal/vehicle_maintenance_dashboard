from utils.model_tool import predict_risk
from utils.retriever import load_retriever


def analyze_vehicle(input_data):
    print("STEP 1: Starting analysis")

    # =========================
    # STEP 1: ML Prediction
    # =========================
    result = predict_risk(input_data)
    risk = result["risk_probability"]

    label = "HIGH" if risk > 0.7 else "MEDIUM" if risk > 0.4 else "LOW"

    print("STEP 2: Building intelligent signals")

    # =========================
    # 🔥 CENTRAL RULE ENGINE
    # =========================
    issues_config = [
        {
            "condition": input_data["oil_temp_avg_celsius"] > 110,
            "label": "Thermal Stress",
            "query": "high engine temperature thermal stress overheating",
            "explanation": "Engine temperature is critically high (>110°C)"
        },
        {
            "condition": input_data["engine_load_percent"] > 80 and input_data["oil_temp_avg_celsius"] > 100,
            "label": "Overloading Stress",
            "query": "thermal stress high engine load overheating",
            "explanation": "High engine load combined with temperature increases wear"
        },
        {
            "condition": input_data["vibration_level"] > 7,
            "label": "Mechanical Vibration",
            "query": "high vibration mechanical imbalance engine wear",
            "explanation": "High vibration detected indicating mechanical imbalance"
        },
        {
            "condition": input_data["battery_voltage"] < 11.5,
            "label": "Battery Failure Risk",
            "query": "low battery voltage electrical failure battery degradation",
            "explanation": "Battery voltage is low (<11.5V)"
        },
        {
            "condition": input_data["fault_code_count"] > 10,
            "label": "System Fault Instability",
            "query": "frequent fault codes system instability failure risk",
            "explanation": "Frequent fault codes detected"
        },
        {
            "condition": input_data["fuel_efficiency_kmpl"] < 10,
            "label": "Low Fuel Efficiency",
            "query": "poor fuel efficiency engine inefficiency performance loss",
            "explanation": "Fuel efficiency is low (<10 kmpl)"
        },
        {
            "condition": input_data["days_since_last_service"] > 180,
            "label": "Delayed Maintenance",
            "query": "delayed maintenance high breakdown risk service overdue",
            "explanation": "Maintenance overdue (>180 days)"
        },
        {
            "condition": input_data["mileage_km"] > 200000,
            "label": "High Vehicle Wear",
            "query": "high mileage vehicle wear and tear aging components",
            "explanation": "Very high mileage indicating wear and tear"
        }
    ]

    # =========================
    # STEP 2: Extract Signals
    # =========================
    signals = []
    clean_issues = []
    explanations = []

    for issue in issues_config:
        if issue["condition"]:
            signals.append(issue["query"])
            clean_issues.append(issue["label"])
            explanations.append(issue["explanation"])

    query = " ".join(signals)

    print("STEP 3: Query ->", query)

    # =========================
    # STEP 3: Retrieve Knowledge
    # =========================
    retriever = load_retriever()
    print("STEP 4: Retriever loaded")

    docs = retriever.invoke(query)

    # 🔥 Smart scoring (better than basic filtering)
    scored_docs = []

    keywords = [w for w in query.split() if w not in ["high", "low", "engine", "stress"]]

    for doc in docs:
        content = doc.page_content.lower()
        score = sum(word in content for word in keywords)
        scored_docs.append((score, doc))

    scored_docs.sort(reverse=True, key=lambda x: x[0])

    filtered_docs = [doc for score, doc in scored_docs if score > 1]

    insights = [doc.page_content for doc in filtered_docs[:3]]

    print("STEP 5: Retrieved relevant insights")

    # =========================
    # STEP 4: Build Action Plan
    # =========================
    action_plan = []

    for insight in insights:
        issue_title = insight.split("\n")[0]

        if risk > 0.7:
            priority = "Immediate"
            timeline = "Within 2 days"
        elif risk > 0.4:
            priority = "High"
            timeline = "Within 1 week"
        else:
            priority = "Moderate"
            timeline = "Monitor"

        action_plan.append({
            "issue": insight,
            "priority": priority,
            "timeline": timeline
        })

    print("STEP 6: Returning result")

    # =========================
    # FINAL OUTPUT
    # =========================
    return {
        "risk_level": label,
        "risk_score": risk,
        "key_issues": clean_issues,
        "explanations": explanations,
        "action_plan": action_plan
    }