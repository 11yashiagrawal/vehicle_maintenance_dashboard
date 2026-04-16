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
    "vehicle_type": "SUV",
    "fuel_type": "Diesel",
    "region": "North",
    "road_condition": "Highway",
    "weather_condition": "Hot",
}


result = analyze_vehicle(sample_input)

assert isinstance(result, dict)
assert "risk_level" in result
assert "risk_score" in result
assert "key_issues" in result
assert "action_plan" in result
assert isinstance(result["action_plan"], list)
assert result["action_plan"], "Action plan should not be empty"

print("test_agents.py passed")
