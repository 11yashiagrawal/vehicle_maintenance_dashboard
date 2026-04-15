from langgraph.graph import StateGraph
from typing import TypedDict

from utils.model_tool import predict_risk
from utils.retriever import load_retriever


# 🔥 Define state (VERY IMPORTANT)
class AgentState(TypedDict):
    input_data: dict
    risk: float
    risk_level: str
    query: str
    insights: list
    final_output: dict


# 🔥 Node 1 — Prediction
def predict_node(state: AgentState):
    result = predict_risk(state["input_data"])

    risk = result["risk_probability"]
    label = "HIGH" if risk > 0.7 else "MEDIUM" if risk > 0.4 else "LOW"

    return {
        "risk": risk,
        "risk_level": label
    }


# 🔥 Node 2 — Build Query
def query_node(state: AgentState):
    data = state["input_data"]
    signals = []

    if data["oil_temp_avg_celsius"] > 110:
        signals.append("high engine temperature")

    if data["vibration_level"] > 7:
        signals.append("high vibration")

    if data["battery_voltage"] < 11.5:
        signals.append("low battery")

    if data["days_since_last_service"] > 180:
        signals.append("delayed maintenance")

    query = " ".join(signals)

    if not query:
        query = "general vehicle maintenance risk"

    return {"query": query}


# 🔥 Node 3 — Retrieval (RAG)
def retrieval_node(state: AgentState):
    retriever = load_retriever()

    docs = retriever.invoke(state["query"])

    insights = [doc.page_content for doc in docs]

    return {"insights": insights}


def decision_node(state: AgentState):
    risk = state["risk"]
    level = state["risk_level"]
    issues = list(set(state["query"].split()))

    # make it readable
    clean_issues = []
    for word in issues:
        if word not in ["high", "low", "engine"]:
            clean_issues.append(word)

    issues = clean_issues

    # 🔥 Health Summary
    health_summary = {
        "Risk Level": level,
        "Risk Score": round(risk, 2),
        "Key Issues": issues
    }

    # 🔥 Action Plan (structured)
    action_plan = []

    for insight in state["insights"]:
        action_plan.append({
            "issue": insight,
            "priority": "Immediate" if level == "HIGH" else "Scheduled",
            "timeline": "Within 2 days" if level == "HIGH" else "Within 1 week"
        })

    return {
        "final_output": {
            "Health Summary": health_summary,
            "Action Plan": action_plan
        }
    }

# 🔥 Build Graph
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("predict", predict_node)
    graph.add_node("query", query_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("decision", decision_node)

    # Flow
    graph.set_entry_point("predict")
    graph.add_edge("predict", "query")
    graph.add_edge("query", "retrieve")
    graph.add_edge("retrieve", "decision")

    return graph.compile()