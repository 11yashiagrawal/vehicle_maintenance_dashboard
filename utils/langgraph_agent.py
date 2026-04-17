from typing import Dict, List, TypedDict

from langgraph.graph import StateGraph

from utils.agent_logic import (
    build_action_plan,
    build_fleet_policy_checks,
    build_query_response,
    build_retrieval_query,
    build_service_outlook,
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
    maintenance_query = extract_user_query(state["input_data"])
    parsed_query_facts = parse_query_facts(maintenance_query)
    merged_input = merge_query_into_input(state["input_data"], parsed_query_facts)
    normalized_input = validate_vehicle_input(merged_input)
    return {
        "normalized_input": normalized_input,
        "maintenance_query": maintenance_query,
        "parsed_query_facts": parsed_query_facts,
    }


def score_node(state: AgentState):
    prediction = predict_risk(state["normalized_input"])
    signals = extract_vehicle_signals(state["normalized_input"], prediction["risk_probability"])
    return {"prediction": prediction, "signals": signals}


def retrieve_node(state: AgentState):
    contexts = retrieve_maintenance_context(state["signals"], state.get("maintenance_query", ""))
    return {"contexts": contexts}


def report_node(state: AgentState):
    prediction = state["prediction"]
    normalized_input = state["normalized_input"]
    action_plan = build_action_plan(normalized_input, state["signals"], state["contexts"])
    action_plan = prioritize_action_plan(action_plan, state.get("maintenance_query", ""))
    service_outlook = build_service_outlook(
        normalized_input,
        state["signals"],
        prediction["risk_probability"],
    )
    fleet_policy_checks = build_fleet_policy_checks(
        normalized_input,
        state["signals"],
        prediction["risk_probability"],
    )
    query_response = build_query_response(
        state.get("maintenance_query", ""),
        action_plan,
        prediction["risk_label"],
        prediction["risk_probability"],
        state["contexts"],
        state.get("parsed_query_facts", {}),
    )

    base_report = {
        "health_summary": {
            "vehicle_status": (
                "Maintenance Required" if prediction["risk_prediction"] == 1 else "No Immediate Maintenance Needed"
            ),
            "risk_level": prediction["risk_label"],
            "risk_score": round(prediction["risk_probability"], 4),
            "key_issues": [signal["issue"] for signal in state["signals"]] or ["no immediate critical issue"],
        },
        "risk_level": prediction["risk_label"],
        "risk_score": round(prediction["risk_probability"], 4),
        "risk_prediction": prediction["risk_prediction"],
        "key_issues": [signal["issue"] for signal in state["signals"]] or ["no immediate critical issue"],
        "maintenance_query": state.get("maintenance_query", ""),
        "parsed_query_facts": state.get("parsed_query_facts", {}),
        "retrieval_mode": get_retriever_mode(),
        "retrieval_query": build_retrieval_query(state["signals"], state.get("maintenance_query", "")),
        "retrieved_context": state["contexts"],
        "sources": state["contexts"],
        "query_response": query_response,
        "service_outlook": service_outlook,
        "fleet_policy_checks": fleet_policy_checks,
        "action_plan": action_plan,
        "disclaimer": (
            "This assistant provides decision support only. For safety-critical faults, stop operating the vehicle "
            "when necessary and consult a certified technician or fleet supervisor before further use."
        ),
    }
    result = maybe_enrich_with_llm(base_report, normalized_input, state["contexts"])
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
