from utils.model_tool import predict_risk
from utils.retriever import load_retriever


def analyze_vehicle(input_data):
    print("STEP 1: Starting analysis")

    # =========================
    # STEP 1: ML Prediction
    # =========================
    result = predict_risk(input_data)
    risk = result["risk_probability"]

    if risk > 0.7:
        label = "HIGH"
    elif risk > 0.4:
        label = "MEDIUM"
    else:
        label = "LOW"

    print("STEP 2: Detecting issues")

    # =========================
    # STEP 2: Detect Issues (Single Source of Truth)
    # =========================
    issues = []

    if input_data["oil_temp_avg_celsius"] > 110:
        issues.append("Thermal Stress")

    if input_data["engine_load_percent"] > 80 and input_data["oil_temp_avg_celsius"] > 100:
        if "Thermal Stress" not in issues:
            issues.append("Thermal Stress")

    if input_data["vibration_level"] > 7:
        issues.append("Mechanical Vibration")

    if input_data["battery_voltage"] < 11.5:
        issues.append("Battery Failure Risk")

    if input_data["fault_code_count"] > 10:
        issues.append("System Fault Instability")

    if input_data["fuel_efficiency_kmpl"] < 10:
        issues.append("Low Fuel Efficiency")

    if input_data["days_since_last_service"] > 180:
        issues.append("Delayed Maintenance")

    if input_data["mileage_km"] > 200000:
        issues.append("High Vehicle Wear")

    print("Detected Issues:", issues)

    # =========================
    # STEP 3: Optional RAG (for enrichment only)
    # =========================
    retriever = load_retriever()
    query = " ".join(issues)

    docs = retriever.invoke(query)
    rag_context = " ".join([doc.page_content for doc in docs[:2]])

    # =========================
    # STEP 4: Build Professional Action Plan
    # =========================
    action_plan = []

    for issue in issues:

        # -------------------------
        # Issue-specific logic
        # -------------------------
        if issue == "Thermal Stress":
            description = f"Engine temperature ({input_data['oil_temp_avg_celsius']}°C) is critically high under load ({input_data['engine_load_percent']}%)"
            action = "Inspect cooling system, radiator, and engine oil immediately"

        elif issue == "Mechanical Vibration":
            description = f"High vibration level detected ({input_data['vibration_level']}) indicating imbalance or wear"
            action = "Check engine mounts, shaft alignment, and rotating components"

        elif issue == "Battery Failure Risk":
            description = f"Battery voltage is low ({input_data['battery_voltage']}V)"
            action = "Test battery health and replace if necessary"

        elif issue == "System Fault Instability":
            description = f"{input_data['fault_code_count']} fault codes detected indicating system instability"
            action = "Run full diagnostics and resolve critical faults"

        elif issue == "Low Fuel Efficiency":
            description = f"Fuel efficiency is low ({input_data['fuel_efficiency_kmpl']} kmpl)"
            action = "Inspect fuel injectors, filters, and engine tuning"

        elif issue == "Delayed Maintenance":
            description = f"Vehicle not serviced for {input_data['days_since_last_service']} days"
            action = "Perform complete preventive maintenance service"

        elif issue == "High Vehicle Wear":
            description = f"High mileage detected ({input_data['mileage_km']} km)"
            action = "Inspect major components and plan replacements"

        else:
            description = "General degradation detected"
            action = "Perform full inspection"

        # -------------------------
        # Context Awareness (🔥 KEY IMPROVEMENT)
        # -------------------------
        context_note = (
            f"Vehicle Type: {input_data['vehicle_type']}, "
            f"Fuel: {input_data['fuel_type']}, "
            f"Operating in {input_data['weather_condition']} weather on "
            f"{input_data['road_condition']} roads increases stress on components."
        )

        # -------------------------
        # Priority Logic
        # -------------------------
        if risk > 0.7:
            priority = "Immediate"
            timeline = "Within 24-48 hours"
        elif risk > 0.4:
            priority = "High"
            timeline = "Within 1 week"
        else:
            priority = "Moderate"
            timeline = "Routine monitoring"

        # -------------------------
        # Final structured issue
        # -------------------------
        action_plan.append({
            "issue": issue,
            "description": description,
            "context": context_note,
            "recommended_action": action,
            "priority": priority,
            "timeline": timeline
        })

    print("STEP 5: Final Output Ready")

    # =========================
    # FINAL OUTPUT
    # =========================
    return {
        "risk_level": label,
        "risk_score": round(risk, 2),
        "key_issues": issues,
        "action_plan": action_plan
    }