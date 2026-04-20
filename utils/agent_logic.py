import json
import logging
import os
import re
from typing import Dict, List

from utils.config import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    FAULT_CODE_ALERT_THRESHOLD,
    HIGH_RISK_THRESHOLD,
    INPUT_CATEGORICAL_FEATURES,
    INPUT_NUMERIC_FEATURES,
    MEDIUM_RISK_THRESHOLD,
    QUERY_FAULT_CODE_BASELINE,
    QUERY_FAULT_CODE_MAX,
    SERVICE_OUTLOOK_IMMEDIATE_THRESHOLD,
    SERVICE_OUTLOOK_SOON_THRESHOLD,
    SERVICE_OVERDUE_DAYS_THRESHOLD,
)
from utils.model_tool import predict_risk
from utils.preprocessor import normalize_input_data
from utils.retriever import get_retriever_mode, load_retriever


logger = logging.getLogger(__name__)


MAINTENANCE_QUERY_PATTERN = re.compile(
    r"\b(vehicle|truck|van|suv|sedan|engine|battery|service|maintenance|fault|code|overheat|"
    r"vibration|brake|steering|diagnostic|oil|cooling|load|efficiency|inspection|repair)\b",
    flags=re.IGNORECASE,
)


def _has_meaningful_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    return bool(value)


def detect_request_mode(input_data: Dict) -> str:
    has_query = _has_meaningful_value(input_data.get("maintenance_query"))
    telemetry_keys = list(INPUT_NUMERIC_FEATURES) + list(INPUT_CATEGORICAL_FEATURES) + ["last_service_date"]
    has_telemetry = any(_has_meaningful_value(input_data.get(key)) for key in telemetry_keys)

    if has_query and has_telemetry:
        return "combined"
    if has_query:
        return "query_only"
    if has_telemetry:
        return "input_only"
    return "empty"


def is_maintenance_query_relevant(maintenance_query: str) -> bool:
    query = str(maintenance_query or "").strip()
    if not query:
        return False
    return bool(MAINTENANCE_QUERY_PATTERN.search(query))


def _query_mentions_overdue_service(query: str) -> bool:
    return bool(
        re.search(r"(service|maintenance)[\w\s]{0,20}(overdue|delayed|missed)", query)
        or re.search(r"(overdue|delayed|missed)[\w\s]{0,20}(service|maintenance)", query)
        or re.search(r"(service|maintenance)[\w\s]{0,20}due", query)
    )


def _query_mentions_fault_codes(query: str) -> bool:
    return bool(
        re.search(r"\bfault\b", query)
        or re.search(r"fault\s*codes?", query)
        or re.search(r"error\s*codes?", query)
        or re.search(r"warning\s*codes?", query)
        or re.search(r"\bdtc\b", query)
        or re.search(r"diagnostic\s*codes?", query)
    )


def validate_vehicle_input(input_data: Dict) -> Dict:
    return normalize_input_data(input_data)


def extract_user_query(input_data: Dict) -> str:
    return str(input_data.get("maintenance_query", "") or "").strip()


def parse_query_facts(maintenance_query: str) -> Dict:
    """
    Extract structured facts from a free-text maintenance query.

    Each inferred value is accompanied by an explanation stored in
    `_inference_log` so callers can surface the reasoning chain in
    decision_trace for transparency (Fix 4).
    """
    query = maintenance_query.lower().strip()
    if not query:
        return {}

    parsed: Dict[str, object] = {}
    inference_log: List[str] = []

    age_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:year|years|yr|yrs)\b", query)
    if age_match:
        parsed["vehicle_age_years"] = float(age_match.group(1))
        inference_log.append(
            f"Inferred vehicle_age_years={age_match.group(1)} from '{age_match.group(0)}' in query."
        )

    vehicle_patterns = {"truck": "Truck", "van": "Van", "suv": "SUV", "sedan": "Sedan"}
    for token, label in vehicle_patterns.items():
        if token in query:
            parsed["vehicle_type"] = label
            inference_log.append(f"Inferred vehicle_type='{label}' from keyword '{token}' in query.")
            break

    weather_patterns = {"hot": "Hot", "heatwave": "Hot", "cold": "Cold", "rain": "Rainy", "rainy": "Rainy"}
    for token, label in weather_patterns.items():
        if token in query:
            parsed["weather_condition"] = label
            inference_log.append(f"Inferred weather_condition='{label}' from keyword '{token}' in query.")
            break

    if any(token in query for token in ["overheat", "hot engine", "high temp", "temperature high", "engine hot"]):
        parsed["oil_temp_avg_celsius"] = max(float(parsed.get("oil_temp_avg_celsius", 0.0)), 115.0)
        inference_log.append(
            "Set oil_temp_avg_celsius=115.0 (above 110C threshold) because query describes overheating symptoms."
        )
    if any(token in query for token in ["vibration", "vibrating", "shake", "shaking", "rattle", "noise"]):
        parsed["vibration_level"] = max(float(parsed.get("vibration_level", 0.0)), 8.0)
        inference_log.append(
            "Set vibration_level=8.0 (above 7.0 threshold) because query describes vibration/noise symptoms."
        )
    if any(token in query for token in ["battery", "not starting", "won't start", "wont start", "dead battery", "starting issue"]):
        parsed["battery_voltage"] = min(float(parsed.get("battery_voltage", 15.0)), 11.2)
        inference_log.append(
            "Set battery_voltage=11.2 (below 11.5V threshold) because query describes starting/electrical weakness."
        )
    if _query_mentions_fault_codes(query):
        parsed["fault_code_count"] = min(
            QUERY_FAULT_CODE_MAX,
            max(float(parsed.get("fault_code_count", 0.0)), QUERY_FAULT_CODE_BASELINE),
        )
        inference_log.append(
            f"Set fault_code_count={parsed['fault_code_count']} (above alert threshold) because query mentions fault/diagnostic codes."
        )
    if _query_mentions_overdue_service(query) or any(token in query for token in ["not serviced", "service delay"]):
        parsed["days_since_last_service"] = max(
            float(parsed.get("days_since_last_service", 0.0)),
            float(SERVICE_OVERDUE_DAYS_THRESHOLD + 10),
        )
        inference_log.append(
            f"Set days_since_last_service={parsed['days_since_last_service']} (overdue proxy) because query mentions delayed/missed service."
        )
    if any(token in query for token in ["high load", "heavy load", "overloaded"]):
        parsed["engine_load_percent"] = max(float(parsed.get("engine_load_percent", 0.0)), 90.0)
        inference_log.append("Set engine_load_percent=90.0 because query describes high/heavy load conditions.")
    if any(token in query for token in ["low mileage", "poor mileage", "fuel efficiency low", "fuel consumption high"]):
        parsed["fuel_efficiency_kmpl"] = min(float(parsed.get("fuel_efficiency_kmpl", 40.0)), 8.0)
        inference_log.append("Set fuel_efficiency_kmpl=8.0 (critically low) because query describes poor fuel efficiency.")
    if "not working" in query or "breakdown" in query:
        parsed["fault_code_count"] = min(
            QUERY_FAULT_CODE_MAX,
            max(float(parsed.get("fault_code_count", 0.0)), QUERY_FAULT_CODE_MAX),
        )
        inference_log.append(
            f"Set fault_code_count={QUERY_FAULT_CODE_MAX} (maximum) because query describes breakdown or complete failure."
        )

    if inference_log:
        parsed["_inference_log"] = inference_log

    return parsed


def merge_query_into_input(input_data: Dict, parsed_query_facts: Dict) -> Dict:
    merged = dict(input_data)

    def has_user_value(field: str) -> bool:
        value = input_data.get(field)
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return True

    for key, value in parsed_query_facts.items():
        if key == "_inference_log":
            continue
        if key == "days_since_last_service":
            if has_user_value("days_since_last_service") or has_user_value("last_service_date"):
                continue
        if key == "fault_code_count":
            if has_user_value("fault_code_count"):
                continue
        if key in {
            "mileage_km", "engine_hours", "vehicle_age_years", "oil_temp_avg_celsius",
            "vibration_level", "battery_voltage", "engine_load_percent", "fuel_efficiency_kmpl",
            "vehicle_type", "fuel_type", "region", "road_condition", "weather_condition",
        } and has_user_value(key):
            continue
        merged[key] = value
    return merged


def extract_vehicle_signals(input_data: Dict, risk_score: float, maintenance_query: str = "") -> List[Dict[str, str]]:
    signals: List[Dict[str, str]] = []
    query = maintenance_query.lower().strip()

    def has_issue(text: str) -> bool:
        return any(text in existing["issue"] for existing in signals)

    if input_data["oil_temp_avg_celsius"] > 110:
        signals.append({
            "issue": "engine overheating risk",
            "reason": "Oil temperature is above 110C, which indicates elevated thermal stress.",
            "priority": "High",
            "timeline": "Inspect immediately",
            "query": "overheating thermal stress engine load",
        })

    if input_data["vibration_level"] > 7:
        signals.append({
            "issue": "abnormal vibration",
            "reason": "High vibration can indicate imbalance, misalignment, or worn components.",
            "priority": "High",
            "timeline": "Inspect within 24 hours",
            "query": "vibration mechanical imbalance worn components",
        })

    if input_data["battery_voltage"] < 11.5:
        signals.append({
            "issue": "battery degradation",
            "reason": "Battery voltage below 11.5V suggests electrical weakness or charging issues.",
            "priority": "Medium",
            "timeline": "Test battery today",
            "query": "battery voltage degradation charging system",
        })

    if input_data["fault_code_count"] > FAULT_CODE_ALERT_THRESHOLD:
        signals.append({
            "issue": "recurring fault activity",
            "reason": "A high number of active fault codes indicates subsystem instability.",
            "priority": "High",
            "timeline": "Run diagnostics this shift",
            "query": "fault codes recurring faults diagnostics",
        })

    if input_data["days_since_last_service"] > SERVICE_OVERDUE_DAYS_THRESHOLD:
        signals.append({
            "issue": "service overdue",
            "reason": "The vehicle has exceeded the preventive maintenance window.",
            "priority": "Medium",
            "timeline": "Schedule service this week",
            "query": "maintenance overdue preventive maintenance service interval",
        })

    if input_data["engine_load_percent"] > 85 and input_data["fuel_efficiency_kmpl"] < 10:
        signals.append({
            "issue": "poor load efficiency",
            "reason": "High load combined with weak efficiency suggests elevated drivetrain stress.",
            "priority": "Medium",
            "timeline": "Inspect in next maintenance cycle",
            "query": "load efficiency drivetrain stress fuel efficiency",
        })

    if query:
        if any(token in query for token in ["overheat", "hot engine", "high temp", "temperature high", "engine hot"]) and not has_issue("overheating"):
            signals.append({
                "issue": "engine overheating risk",
                "reason": "The maintenance query describes overheating symptoms that require cooling-system checks.",
                "priority": "High",
                "timeline": "Inspect immediately",
                "query": "overheating thermal stress engine load",
            })
        if any(token in query for token in ["vibration", "shake", "shaking", "rattle", "noise"]) and not has_issue("vibration"):
            signals.append({
                "issue": "abnormal vibration",
                "reason": "The query mentions vibration/noise symptoms commonly linked to rotating or mounting components.",
                "priority": "High",
                "timeline": "Inspect within 24 hours",
                "query": "vibration mechanical imbalance worn components",
            })
        if any(token in query for token in ["battery", "not starting", "won't start", "wont start", "dead battery"]) and not has_issue("battery"):
            signals.append({
                "issue": "battery degradation",
                "reason": "The query indicates starting/electrical weakness, so battery and charging checks should be prioritized.",
                "priority": "Medium",
                "timeline": "Test battery today",
                "query": "battery voltage degradation charging system",
            })
        if _query_mentions_fault_codes(query) and not has_issue("fault"):
            signals.append({
                "issue": "recurring fault activity",
                "reason": "The query references fault/diagnostic codes, suggesting unresolved subsystem alerts.",
                "priority": "High",
                "timeline": "Run diagnostics this shift",
                "query": "fault codes recurring faults diagnostics",
            })
        if (_query_mentions_overdue_service(query) or any(token in query for token in ["not serviced", "service delay"])) and not has_issue("service overdue"):
            signals.append({
                "issue": "service overdue",
                "reason": "The query indicates delayed preventive maintenance and elevated wear risk.",
                "priority": "Medium",
                "timeline": "Schedule service this week",
                "query": "maintenance overdue preventive maintenance service interval",
            })
        if any(token in query for token in ["brake", "braking", "stopping distance"]) and not has_issue("brake"):
            signals.append({
                "issue": "brake system concern",
                "reason": "The query mentions braking behavior; safety checks on brake pads, fluid, and hydraulics are required.",
                "priority": "High",
                "timeline": "Inspect immediately",
                "query": "brake inspection stopping distance hydraulic leak",
            })
        if any(token in query for token in ["steering", "pulling", "alignment"]) and not has_issue("steering"):
            signals.append({
                "issue": "steering or alignment concern",
                "reason": "The query suggests steering/alignment symptoms that may affect handling safety.",
                "priority": "Medium",
                "timeline": "Inspect within 24 hours",
                "query": "steering alignment suspension handling",
            })

    if risk_score >= HIGH_RISK_THRESHOLD and not signals:
        signals.append({
            "issue": "high predicted maintenance risk",
            "reason": "The ML model predicts high failure risk even though no single threshold rule dominates.",
            "priority": "High",
            "timeline": "Inspect within 24 hours",
            "query": "high risk preventive inspection diagnostics",
        })

    return signals


def build_retrieval_query(signals: List[Dict[str, str]], maintenance_query: str) -> str:
    signal_query = " ".join(signal["query"] for signal in signals).strip()
    if maintenance_query and signal_query:
        return f"{maintenance_query} {signal_query}"
    if maintenance_query:
        return maintenance_query
    if signal_query:
        return signal_query
    return "preventive maintenance vehicle diagnostics"


def retrieve_maintenance_context(signals: List[Dict[str, str]], maintenance_query: str = "") -> List[str]:
    query = build_retrieval_query(signals, maintenance_query)
    retriever = load_retriever()
    docs = retriever.invoke(query)
    return [doc.page_content for doc in docs]


def _find_best_context_for_signal(signal_issue: str, contexts: List[str]) -> str:
    """Semantically match retrieved context to a signal by keyword overlap."""
    issue_terms = set(signal_issue.lower().replace("_", " ").split())
    best_score, best_ctx = -1, contexts[0] if contexts else ""
    for ctx in contexts:
        ctx_lower = ctx.lower()
        score = sum(1 for t in issue_terms if t in ctx_lower)
        score += sum(2 for t in issue_terms if t in ctx_lower.split("\n")[0])
        if score > best_score:
            best_score, best_ctx = score, ctx
    for line in best_ctx.split("\n"):
        line = line.strip()
        if len(line) > 30 and not line.isupper():
            return line[:240]
    return best_ctx[:240]


def build_action_plan(input_data: Dict, signals: List[Dict[str, str]], contexts: List[str]) -> List[Dict[str, str]]:
    """
    Build a structured action plan from detected signals.

    Each item gets a `supporting_guideline` that is semantically matched
    to the signal issue from retrieved context (Fix 5: replaces fixed slice).
    """
    action_plan: List[Dict[str, str]] = []

    for signal in signals:
        issue = signal["issue"]
        if "overheating" in issue:
            action = (
                "Check cooling system (coolant level, radiator, thermostat), "
                "oil condition, and sustained engine load before further long trips."
            )
        elif "vibration" in issue:
            action = (
                "Inspect wheel balance, tyre condition, CV joints, driveshaft, "
                "engine mounts, and transmission mounts."
            )
        elif "battery" in issue:
            action = (
                "Conduct a battery load test. Replace if capacity is below 70% "
                "of rated CCA. Check alternator output and terminal condition."
            )
        elif "fault" in issue:
            action = (
                "Scan and group diagnostic codes by subsystem. Address recurring "
                "faults before clearing codes — do not mask underlying failures."
            )
        elif "service overdue" in issue:
            action = (
                "Perform scheduled preventive maintenance: oil change, filter "
                "replacement, brake inspection, tyre pressure, and fluid levels."
            )
        elif "brake" in issue:
            action = (
                "Inspect brake pad thickness, fluid condition, and hydraulic lines. "
                "Check for ABS fault codes. Ground vehicle if stopping distance is affected."
            )
        elif "steering" in issue:
            action = (
                "Check wheel alignment, tie rod ends, ball joints, and control arm "
                "bushings. Inspect for suspension noise or handling pull."
            )
        elif "load efficiency" in issue:
            action = (
                "Inspect drivetrain under load. Check fuel injectors, air filter, "
                "and transmission fluid condition."
            )
        else:
            action = "Run a full inspection and prioritize any failing subsystem identified in diagnostics."

        action_plan.append({
            "issue": issue.title(),
            "reason": signal["reason"],
            "context_impact": (
                f"{input_data['vehicle_type']} operating in {input_data['weather_condition'].lower()} "
                f"weather on {input_data['road_condition'].lower()} roads increases the relevance of this issue."
            ),
            "action": action,
            "priority": signal["priority"],
            "timeline": signal["timeline"],
            "supporting_guideline": _find_best_context_for_signal(issue, contexts) if contexts else "",
        })

    if not action_plan:
        action_plan.append({
            "issue": "Preventive Monitoring",
            "reason": "No severe rule-based issue was detected, but the vehicle should continue routine monitoring.",
            "context_impact": (
                f"The current operating profile for this {input_data['vehicle_type'].lower()} does not suggest an immediate fault."
            ),
            "action": (
                "Continue scheduled inspections and monitor telemetry for sudden changes. "
                "Next service within the standard interval."
            ),
            "priority": "Low",
            "timeline": "Next service cycle",
            "supporting_guideline": contexts[0][:240] if contexts else
                "Preventive maintenance reduces long-term operational risk.",
        })

    return action_plan


def prioritize_action_plan(action_plan: List[Dict[str, str]], maintenance_query: str) -> List[Dict[str, str]]:
    query = maintenance_query.lower().strip()
    if not query:
        return action_plan

    ranked: List[tuple[int, Dict[str, str]]] = []
    for item in action_plan:
        searchable = " ".join([
            item.get("issue", ""),
            item.get("reason", ""),
            item.get("action", ""),
        ]).lower()
        score = sum(term in searchable for term in query.split())
        ranked.append((score, item))

    ranked.sort(key=lambda row: row[0], reverse=True)
    return [item for _, item in ranked]


def build_service_outlook(input_data: Dict, signals: List[Dict[str, str]], risk_score: float) -> Dict[str, str]:
    if risk_score >= SERVICE_OUTLOOK_IMMEDIATE_THRESHOLD:
        inspection_window = "Immediate workshop intake"
        downtime_risk = "High"
        operating_advice = "Avoid long-haul or passenger-critical duty until inspection is completed."
    elif risk_score >= SERVICE_OUTLOOK_SOON_THRESHOLD:
        inspection_window = "Within 48 hours"
        downtime_risk = "Moderate"
        operating_advice = "Allow restricted operations and monitor telemetry between trips."
    else:
        inspection_window = "Next planned maintenance window"
        downtime_risk = "Low"
        operating_advice = "Vehicle can remain in service with routine monitoring."

    focus_areas = []
    if any("overheating" in signal["issue"] for signal in signals):
        focus_areas.append("cooling system and oil health")
    if any("vibration" in signal["issue"] for signal in signals):
        focus_areas.append("mounts, wheels, and drivetrain alignment")
    if any("battery" in signal["issue"] for signal in signals):
        focus_areas.append("battery and charging system")
    if any("fault" in signal["issue"] for signal in signals):
        focus_areas.append("ECU diagnostics and recurring fault clusters")
    if not focus_areas:
        focus_areas.append("routine preventive inspection")

    return {
        "inspection_window": inspection_window,
        "downtime_risk": downtime_risk,
        "operating_advice": operating_advice,
        "primary_focus": ", ".join(focus_areas[:3]),
    }


def build_fleet_policy_checks(input_data: Dict, signals: List[Dict[str, str]], risk_score: float) -> List[Dict[str, str]]:
    checks: List[Dict[str, str]] = []

    age = float(input_data.get("vehicle_age_years", 0))
    days_since_service = int(input_data.get("days_since_last_service", 0))
    vehicle_type = str(input_data.get("vehicle_type", "Vehicle"))
    road_condition = str(input_data.get("road_condition", "Urban"))
    weather = str(input_data.get("weather_condition", "Normal"))

    if age >= 7 and vehicle_type in {"Truck", "Van", "SUV"}:
        checks.append({
            "title": "Lifecycle Review Trigger",
            "status": "Attention",
            "detail": (
                f"{vehicle_type} units older than 7 years should enter a fleet lifecycle review for deep inspection, "
                "major component replacement planning, or phased retirement assessment."
            ),
        })

    if days_since_service > SERVICE_OVERDUE_DAYS_THRESHOLD:
        checks.append({
            "title": "Preventive Maintenance SLA Breach",
            "status": "Critical",
            "detail": "The vehicle is beyond the recommended service interval, so it should be prioritized in the maintenance queue.",
        })

    if any("recurring fault activity" == signal["issue"] for signal in signals):
        checks.append({
            "title": "Diagnostic Escalation",
            "status": "Critical",
            "detail": "Repeated fault activity should be escalated from basic service to root-cause diagnostics before the vehicle returns to full duty.",
        })

    if road_condition == "Highway" and risk_score >= HIGH_RISK_THRESHOLD:
        checks.append({
            "title": "Route Assignment Restriction",
            "status": "Attention",
            "detail": "High-risk vehicles should be moved off long continuous highway duty until inspection confirms stable operating health.",
        })

    if weather in {"Hot", "Cold"} and risk_score >= MEDIUM_RISK_THRESHOLD:
        checks.append({
            "title": "Weather Sensitivity Alert",
            "status": "Monitor",
            "detail": (
                f"{weather} weather increases stress on temperature-sensitive and electrical systems, so telemetry should be reviewed more frequently."
            ),
        })

    return checks


def build_query_response(
    maintenance_query: str,
    action_plan: List[Dict[str, str]],
    risk_label: str,
    risk_score: float,
    contexts: List[str],
    parsed_query_facts: Dict,
) -> Dict:
    if not maintenance_query.strip() or not is_maintenance_query_relevant(maintenance_query):
        return {}

    prioritized_plan = prioritize_action_plan(action_plan, maintenance_query)
    top_actions = prioritized_plan[:3]
    first_action = top_actions[0] if top_actions else {}

    if risk_label == "HIGH":
        short_answer = "Yes, this case should be treated as urgent. Move the vehicle toward inspection and address the highest-risk subsystem first."
    elif risk_label == "MEDIUM":
        short_answer = "This does not look critical yet, but it should be inspected soon before the issue becomes operationally expensive."
    else:
        short_answer = "The current risk is low, so continue service with targeted checks and normal monitoring."

    immediate_steps = [
        {
            "step": item.get("action", "Inspect the vehicle."),
            "priority": item.get("priority", "Unknown"),
            "timeline": item.get("timeline", "Not specified"),
        }
        for item in top_actions
    ]

    evidence = []
    for context in contexts[:2]:
        first_line = context.splitlines()[0].strip()
        evidence.append(first_line if first_line else "Maintenance guidance")

    public_facts = {k: v for k, v in parsed_query_facts.items() if k != "_inference_log"}

    return {
        "question": maintenance_query,
        "short_answer": short_answer,
        "recommended_focus": first_action.get("issue", "General inspection"),
        "immediate_steps": immediate_steps,
        "evidence_used": evidence,
        "risk_context": f"{risk_label.title()} risk ({risk_score:.1%})",
        "parsed_facts": public_facts,
    }


def merge_report(base_report: Dict, enriched: Dict) -> Dict:
    """Safely merge enriched report, validating action_plan and critical fields."""
    merged = dict(base_report)
    for key, value in enriched.items():
        if key == "action_plan":
            if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
                logger.warning("Skipping malformed action_plan from enrichment.")
                continue
        if isinstance(value, dict) and isinstance(base_report.get(key), dict):
            nested = dict(base_report[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def maybe_enrich_with_llm(base_report: Dict, input_data: Dict, contexts: List[str]) -> Dict:
    if os.getenv("ENABLE_OLLAMA_ENRICHMENT", "0") != "1":
        return base_report

    try:
        from langchain_ollama import ChatOllama
    except Exception:
        logger.debug("langchain_ollama not available; skipping LLM enrichment.")
        return base_report

    ollama_model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)

    llm = ChatOllama(model=ollama_model, base_url=ollama_base_url, temperature=0, format="json")
    prompt = f"""
You are a vehicle maintenance copilot.
Return valid JSON only.
Keep the same top-level keys and do not invent unsupported issues.

INPUT:
{json.dumps(input_data, default=str)}

CONTEXT:
{json.dumps(contexts, default=str)}

BASE_REPORT:
{json.dumps(base_report, default=str)}
"""

    try:
        response = llm.invoke(prompt)
        content = response.content if isinstance(response.content, str) else json.dumps(response.content)
        enriched = json.loads(content)
        if isinstance(enriched, dict):
            ap = enriched.get("action_plan", [])
            if "action_plan" in enriched:
                if not (isinstance(ap, list) and all(isinstance(item, dict) for item in ap)):
                    logger.warning("Enrichment action_plan failed shape validation; using base report.")
                    return base_report
            return merge_report(base_report, enriched)
    except Exception as exc:
        logger.warning("LLM enrichment failed: %s", exc)
        base_report["enrichment_status"] = "failed"
        return base_report

    return base_report


def analyze_vehicle(input_data: Dict) -> Dict:
    """
    Standalone analysis pipeline (used for testing and fallback).
    For production, prefer the LangGraph agent in langgraph_agent.py.
    """
    request_mode = detect_request_mode(input_data)
    maintenance_query = extract_user_query(input_data)
    query_is_relevant = is_maintenance_query_relevant(maintenance_query)
    parsed_query_facts = parse_query_facts(maintenance_query)
    merged_input = merge_query_into_input(input_data, parsed_query_facts)
    normalized_input = validate_vehicle_input(merged_input)
    prediction = predict_risk(normalized_input)
    risk_score = prediction["risk_probability"]
    risk_label = prediction["risk_label"]

    signals = extract_vehicle_signals(normalized_input, risk_score, maintenance_query)
    contexts = retrieve_maintenance_context(signals, maintenance_query)
    action_plan = build_action_plan(normalized_input, signals, contexts)
    action_plan = prioritize_action_plan(action_plan, maintenance_query)
    service_outlook = build_service_outlook(normalized_input, signals, risk_score)
    fleet_policy_checks = build_fleet_policy_checks(normalized_input, signals, risk_score)
    query_response = build_query_response(
        maintenance_query, action_plan, risk_label, risk_score, contexts, parsed_query_facts,
    )

    # Surface query inference log in decision_trace for transparency (Fix 4)
    inference_log = parsed_query_facts.get("_inference_log", [])
    public_facts = {k: v for k, v in parsed_query_facts.items() if k != "_inference_log"}

    report = {
        "request_mode": request_mode,
        "query_is_relevant": query_is_relevant,
        "health_summary": {
            "vehicle_status": (
                "Maintenance Required" if prediction["risk_prediction"] == 1 else "No Immediate Maintenance Needed"
            ),
            "risk_level": risk_label,
            "risk_score": round(risk_score, 4),
            "key_issues": [signal["issue"] for signal in signals] or ["no immediate critical issue"],
        },
        "risk_level": risk_label,
        "risk_score": round(risk_score, 4),
        "risk_prediction": prediction["risk_prediction"],
        "key_issues": [signal["issue"] for signal in signals] or ["no immediate critical issue"],
        "maintenance_query": maintenance_query,
        "parsed_query_facts": public_facts,
        "retrieval_mode": get_retriever_mode(),
        "retrieval_query": build_retrieval_query(signals, maintenance_query),
        "retrieved_context": contexts,
        "sources": contexts,
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
            "detected_signals": [signal["issue"] for signal in signals],
            "retrieval_query": build_retrieval_query(signals, maintenance_query),
            "retrieval_mode": get_retriever_mode(),
            "top_recommendation": action_plan[0]["issue"] if action_plan else "Preventive Monitoring",
        },
        "disclaimer": (
            "This assistant provides decision support only. For safety-critical faults, stop operating the vehicle "
            "when necessary and consult a certified technician or fleet supervisor before further use."
        ),
    }

    return maybe_enrich_with_llm(report, normalized_input, contexts)
