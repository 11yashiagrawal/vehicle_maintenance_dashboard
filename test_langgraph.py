from utils.langgraph_agent import build_agent
import json


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
    "vehicle_type": "SUV",
    "fuel_type": "Diesel",
    "region": "North",
    "road_condition": "Highway",
    "weather_condition": "Hot",
}

result = agent.invoke({"input_data": input_data})

assert "result" in result
assert isinstance(result["result"], dict)
assert "action_plan" in result["result"]
assert result["result"]["action_plan"], "Graph should return a non-empty action plan"

print("\n=== LANGGRAPH AGENT OUTPUT ===\n")
print(json.dumps(result["result"], indent=2))
print("test_langgraph.py passed")
