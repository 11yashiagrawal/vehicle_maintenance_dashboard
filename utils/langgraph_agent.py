from typing import Dict, List, TypedDict

from langgraph.graph import StateGraph

from utils.agent_logic import (
    build_action_plan,
    build_fleet_policy_checks,
    build_query_response,
    build_retrieval_query,
    build_service_outlook,
    detect_request_mode,
    extract_vehicle_signals,
    extract_user_query,
    merge_query_into_input,
    maybe_enrich_with_llm,
    parse_query_facts,
    prioritize_action_plan,
    retrieve_maintenance_context,
    validate_vehicle_input,
)
from utils.model_tool import predict_risk
from utils.retriever import get_retriever_mode


class AgentState(TypedDict, total=False):
    input_data: Dict
    normalized_input: Dict
    maintenance_query: str
    parsed_query_facts: Dict
    prediction: Dict
    signals: List[Dict[str, str]]
    contexts: List[str]
    result: Dict


def validate_node(state: AgentState):
    input_data = state.get("input_data", {})
    maintenance_query = extract_user_query(input_data)
    parsed_query_facts = parse_query_facts(maintenance_query)
    merged_input = merge_query_into_input(input_data, parsed_query_facts)
    normalized_input = validate_vehicle_input(merged_input)
    return {
        "normalized_input": normalized_input,
        "maintenance_query": maintenance_query,
        "parsed_query_facts": parsed_query_facts,
    }


def score_node(state: AgentState):
    normalized_input = state.get("normalized_input", {})
    prediction = predict_risk(normalized_input)
    signals = extract_vehicle_signals(
        normalized_input,
        prediction["risk_probability"],
        state.get("maintenance_query", ""),
    )
    return {"prediction": prediction, "signals": signals}


def retrieve_node(state: AgentState):
    signals = state.get("signals", [])
    contexts = retrieve_maintenance_context(signals, state.get("maintenance_query", ""))
    return {"contexts": contexts}


def report_node(state: AgentState):
    prediction = state.get("prediction", {})
    normalized_input = state.get("normalized_input", {})
    signals = state.get("signals", [])
    contexts = state.get("contexts", [])
    request_mode = detect_request_mode(state.get("input_data", {}))

    action_plan = build_action_plan(normalized_input, signals, contexts)
    action_plan = prioritize_action_plan(action_plan, state.get("maintenance_query", ""))
    service_outlook = build_service_outlook(
        normalized_input,
        signals,
        float(prediction.get("risk_probability", 0.0)),
    )
    fleet_policy_checks = build_fleet_policy_checks(
        normalized_input,
        signals,
        float(prediction.get("risk_probability", 0.0)),
    )
    query_response = build_query_response(
        state.get("maintenance_query", ""),
        action_plan,
        str(prediction.get("risk_label", "LOW")),
        float(prediction.get("risk_probability", 0.0)),
        contexts,
        state.get("parsed_query_facts", {}),
    )

    base_report = {
        "request_mode": request_mode,
        "health_summary": {
            "vehicle_status": (
                "Maintenance Required" if int(prediction.get("risk_prediction", 0)) == 1 else "No Immediate Maintenance Needed"
            ),
            "risk_level": str(prediction.get("risk_label", "LOW")),
            "risk_score": round(float(prediction.get("risk_probability", 0.0)), 4),
            "key_issues": [signal["issue"] for signal in signals] or ["no immediate critical issue"],
        },
        "risk_level": str(prediction.get("risk_label", "LOW")),
        "risk_score": round(float(prediction.get("risk_probability", 0.0)), 4),
        "risk_prediction": int(prediction.get("risk_prediction", 0)),
        "key_issues": [signal["issue"] for signal in signals] or ["no immediate critical issue"],
        "maintenance_query": state.get("maintenance_query", ""),
        "parsed_query_facts": state.get("parsed_query_facts", {}),
        "retrieval_mode": get_retriever_mode(),
        "retrieval_query": build_retrieval_query(signals, state.get("maintenance_query", "")),
        "retrieved_context": contexts,
        "sources": contexts,
        "query_response": query_response,
        "service_outlook": service_outlook,
        "fleet_policy_checks": fleet_policy_checks,
        "action_plan": action_plan,
        "decision_trace": {
            "request_mode": request_mode,
            "query_present": bool(str(state.get("maintenance_query", "")).strip()),
            "telemetry_present": request_mode in {"input_only", "combined"},
            "parsed_query_facts": state.get("parsed_query_facts", {}),
            "detected_signals": [signal["issue"] for signal in signals],
            "retrieval_query": build_retrieval_query(signals, state.get("maintenance_query", "")),
            "retrieval_mode": get_retriever_mode(),
            "top_recommendation": action_plan[0]["issue"] if action_plan else "Preventive Monitoring",
        },
        "disclaimer": (
            "This assistant provides decision support only. For safety-critical faults, stop operating the vehicle "
            "when necessary and consult a certified technician or fleet supervisor before further use."
        ),
    }
    result = maybe_enrich_with_llm(base_report, normalized_input, contexts)
    return {"result": result}


def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("validate", validate_node)
    graph.add_node("score", score_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("validate")
    graph.add_edge("validate", "score")
    graph.add_edge("score", "retrieve")
    graph.add_edge("retrieve", "report")
    graph.set_finish_point("report")

    return graph.compile()
