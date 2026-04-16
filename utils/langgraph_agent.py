from typing import Dict, List, TypedDict

from langgraph.graph import StateGraph

from utils.agent_logic import (
    build_action_plan,
    extract_vehicle_signals,
    maybe_enrich_with_llm,
    retrieve_maintenance_context,
    validate_vehicle_input,
)
from utils.model_tool import predict_risk


class AgentState(TypedDict, total=False):
    input_data: Dict
    normalized_input: Dict
    prediction: Dict
    signals: List[Dict[str, str]]
    contexts: List[str]
    result: Dict


def validate_node(state: AgentState):
    normalized_input = validate_vehicle_input(state["input_data"])
    return {"normalized_input": normalized_input}


def score_node(state: AgentState):
    prediction = predict_risk(state["normalized_input"])
    signals = extract_vehicle_signals(state["normalized_input"], prediction["risk_probability"])
    return {"prediction": prediction, "signals": signals}


def retrieve_node(state: AgentState):
    contexts = retrieve_maintenance_context(state["signals"])
    return {"contexts": contexts}


def report_node(state: AgentState):
    prediction = state["prediction"]
    normalized_input = state["normalized_input"]
    action_plan = build_action_plan(normalized_input, state["signals"], state["contexts"])

    base_report = {
        "risk_level": prediction["risk_label"],
        "risk_score": round(prediction["risk_probability"], 4),
        "risk_prediction": prediction["risk_prediction"],
        "key_issues": [signal["issue"] for signal in state["signals"]] or ["no immediate critical issue"],
        "retrieved_context": state["contexts"],
        "action_plan": action_plan,
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
