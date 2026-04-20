"""
utils/langgraph_agent.py

LangGraph agent with conditional routing (Fix 2):
  - After score_node, routes HIGH-risk vehicles through full retrieve → report.
  - LOW/MEDIUM-risk, query-only cases skip straight to a lightweight report node,
    saving retrieval overhead and demonstrating LangGraph conditional branching.

Other improvements vs original:
  - AgentState carries `sources` and `retrieval_query` from retriever metadata.
  - build_action_plan uses semantic context matching per signal (Fix 5).
  - decision_trace surfaces query_inference_log (Fix 4).
  - analyze_vehicle() in agent_logic.py is the fallback; this graph is production path (Fix 3).
"""

from typing import Dict, List, TypedDict

from langgraph.graph import StateGraph

from utils.agent_logic import (
    build_fleet_policy_checks,
    build_query_response,
    build_retrieval_query,
    build_service_outlook,
    detect_request_mode,
    extract_vehicle_signals,
    extract_user_query,
    is_maintenance_query_relevant,
    maybe_enrich_with_llm,
    merge_query_into_input,
    parse_query_facts,
    prioritize_action_plan,
    _find_best_context_for_signal,
    validate_vehicle_input,
)
from utils.config import HIGH_RISK_THRESHOLD
from utils.model_tool import predict_risk
from utils.retriever import get_retriever_mode, load_retriever


# ---------------------------------------------------------------------------
# Patched helpers
# ---------------------------------------------------------------------------

def _retrieve_with_sources(signals, maintenance_query=""):
    """Return contexts, source labels, and the retrieval query used."""
    query = build_retrieval_query(signals, maintenance_query)
    retriever = load_retriever()
    docs = retriever.invoke(query)
    contexts = [doc.page_content for doc in docs]
    sources = [doc.metadata.get("source", "maintenance_docs.txt") for doc in docs]
    return contexts, sources, query


def _build_action_plan(input_data, signals, contexts):
    """
    Semantic action plan builder — each item gets a context snippet matched
    to its signal rather than a fixed slice of the first document.
    """
    from utils.agent_logic import build_action_plan
    return build_action_plan(input_data, signals, contexts)


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    input_data: Dict
    normalized_input: Dict
    maintenance_query: str
    parsed_query_facts: Dict
    prediction: Dict
    signals: List[Dict[str, str]]
    contexts: List[str]
    sources: List[str]
    retrieval_query: str
    result: Dict


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

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


def route_after_score(state: AgentState) -> str:
    """
    Conditional edge (Fix 2): route HIGH-risk vehicles through full RAG retrieval.
    LOW/MEDIUM cases with no signals go to a lightweight report.
    """
    prediction = state.get("prediction", {})
    signals = state.get("signals", [])
    risk_prob = float(prediction.get("risk_probability", 0.0))

    # Always retrieve if: high risk OR any signals found OR there's a user query
    maintenance_query = state.get("maintenance_query", "")
    if risk_prob >= HIGH_RISK_THRESHOLD or signals or maintenance_query.strip():
        return "retrieve"
    return "report_lite"


def retrieve_node(state: AgentState):
    """Full RAG retrieval — used for high-risk or signal-bearing cases."""
    signals = state.get("signals", [])
    maintenance_query = state.get("maintenance_query", "")
    contexts, sources, retrieval_query = _retrieve_with_sources(signals, maintenance_query)
    return {
        "contexts": contexts,
        "sources": sources,
        "retrieval_query": retrieval_query,
    }


def _assemble_report(state: AgentState, contexts: List[str], sources: List[str], retrieval_query: str) -> Dict:
    """Shared report assembly used by both report_node and report_lite_node."""
    prediction = state.get("prediction", {})
    normalized_input = state.get("normalized_input", {})
    signals = state.get("signals", [])
    request_mode = detect_request_mode(state.get("input_data", {}))
    query_is_relevant = is_maintenance_query_relevant(state.get("maintenance_query", ""))
    maintenance_query = state.get("maintenance_query", "")
    parsed_query_facts = state.get("parsed_query_facts", {})

    action_plan = _build_action_plan(normalized_input, signals, contexts)
    action_plan = prioritize_action_plan(action_plan, maintenance_query)

    service_outlook = build_service_outlook(
        normalized_input, signals, float(prediction.get("risk_probability", 0.0))
    )
    fleet_policy_checks = build_fleet_policy_checks(
        normalized_input, signals, float(prediction.get("risk_probability", 0.0))
    )
    query_response = build_query_response(
        maintenance_query,
        action_plan,
        str(prediction.get("risk_label", "LOW")),
        float(prediction.get("risk_probability", 0.0)),
        contexts,
        parsed_query_facts,
    )

    inference_log = parsed_query_facts.get("_inference_log", [])
    public_facts = {k: v for k, v in parsed_query_facts.items() if k != "_inference_log"}

    base_report = {
        "request_mode": request_mode,
        "query_is_relevant": query_is_relevant,
        "health_summary": {
            "vehicle_status": (
                "Maintenance Required"
                if int(prediction.get("risk_prediction", 0)) == 1
                else "No Immediate Maintenance Needed"
            ),
            "risk_level": str(prediction.get("risk_label", "LOW")),
            "risk_score": round(float(prediction.get("risk_probability", 0.0)), 4),
            "key_issues": [s["issue"] for s in signals] or ["no immediate critical issue"],
        },
        "risk_level": str(prediction.get("risk_label", "LOW")),
        "risk_score": round(float(prediction.get("risk_probability", 0.0)), 4),
        "risk_prediction": int(prediction.get("risk_prediction", 0)),
        "key_issues": [s["issue"] for s in signals] or ["no immediate critical issue"],
        "maintenance_query": maintenance_query,
        "parsed_query_facts": public_facts,
        "retrieval_mode": get_retriever_mode(),
        "retrieval_query": retrieval_query,
        "retrieved_context": contexts,
        "sources": sources,
        "query_response": query_response,
        "service_outlook": service_outlook,
        "fleet_policy_checks": fleet_policy_checks,
        "action_plan": action_plan,
        "decision_trace": {
            "request_mode": request_mode,
            "query_present": bool(maintenance_query.strip()),
            "query_is_relevant": query_is_relevant,
            "telemetry_present": request_mode in {"input_only", "combined"},
            "parsed_query_facts": public_facts,
            "query_inference_log": inference_log,
            "detected_signals": [s["issue"] for s in signals],
            "retrieval_query": retrieval_query,
            "retrieval_mode": get_retriever_mode(),
            "sources": sources,
            "routing_path": "retrieve→report" if contexts else "report_lite",
            "top_recommendation": action_plan[0]["issue"] if action_plan else "Preventive Monitoring",
        },
        "disclaimer": (
            "This assistant provides decision support only. For safety-critical faults, "
            "stop operating the vehicle when necessary and consult a certified technician "
            "or fleet supervisor before further use."
        ),
    }

    return maybe_enrich_with_llm(base_report, normalized_input, contexts)


def report_node(state: AgentState):
    """Full report — called after retrieval for high-risk / signal-bearing cases."""
    contexts = state.get("contexts", [])
    sources = state.get("sources", [])
    retrieval_query = state.get("retrieval_query", "")
    result = _assemble_report(state, contexts, sources, retrieval_query)
    return {"result": result}


def report_lite_node(state: AgentState):
    """
    Lightweight report — skips retrieval for low-risk, no-signal, no-query cases.
    Demonstrates LangGraph conditional branching (Fix 2).
    """
    fallback_context = [
        "Preventive maintenance baseline: check oil quality, cooling health, and brake condition every service cycle."
    ]
    result = _assemble_report(state, fallback_context, [], "")
    return {"result": result}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("validate", validate_node)
    graph.add_node("score", score_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("report", report_node)
    graph.add_node("report_lite", report_lite_node)

    graph.set_entry_point("validate")
    graph.add_edge("validate", "score")

    # Conditional edge: HIGH-risk/signals → retrieve → report
    #                   LOW-risk/no signals → report_lite directly
    graph.add_conditional_edges(
        "score",
        route_after_score,
        {
            "retrieve": "retrieve",
            "report_lite": "report_lite",
        },
    )

    graph.add_edge("retrieve", "report")
    graph.set_finish_point("report")
    graph.set_finish_point("report_lite")

    return graph.compile()
