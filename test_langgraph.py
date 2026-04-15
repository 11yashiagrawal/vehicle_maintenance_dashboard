from utils.langgraph_agent import build_agent

agent = build_agent()

input_data = {
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

result = agent.invoke({
    "input_data": input_data
})

output = result["final_output"]

print("\n=== VEHICLE HEALTH REPORT ===\n")

# Health Summary
print("🔍 Health Summary")
for key, value in output["Health Summary"].items():
    print(f"{key}: {value}")

print("\n Action Plan")

for i, item in enumerate(output["Action Plan"], 1):
    print(f"\n{i}. Issue:")
    print(item["issue"])
    print(f"   Priority: {item['priority']}")
    print(f"   Timeline: {item['timeline']}")