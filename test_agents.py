from utils.agent_logic import analyze_vehicle

sample_input = {
    "mileage_km": 120000,
    "engine_hours": 4000,
    "vehicle_age_years": 6,
    "fault_code_count": 12,
    "oil_temp_avg_celsius": 115,
    "vibration_level": 8,
    "battery_voltage": 10.5,
    "engine_load_percent": 90,
    "fuel_efficiency_kmpl": 8,
    "days_since_last_service": 200,
    "vehicle_type": "Bus",
    "fuel_type": "Diesel",
    "region": "North",
    "road_condition": "Rural",
    "weather_condition": "Hot"
}

result = analyze_vehicle(sample_input)

print("\n=== VEHICLE HEALTH REPORT ===")

print("\n🔍 Health Summary")
print("Risk Level:", result["risk_level"])
print("Risk Score:", result["risk_score"])
print("Key Issues:", result["key_issues"])

print("\n Action Plan")

for i, item in enumerate(result["action_plan"], 1):
    print(f"\n{i}. 🔧 {item['issue'].splitlines()[0]}")

    description = "\n".join(item["issue"].splitlines()[1:])
    print(description)

    print(f"\n   Priority: {item['priority']}")
    print(f"   Timeline: {item['timeline']}")