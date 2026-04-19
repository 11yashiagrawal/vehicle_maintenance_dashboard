import streamlit as st
from datetime import date

from utils.langgraph_agent import build_agent
from utils.agent_logic import is_maintenance_query_relevant
from utils.model_loader import load_artifacts
from utils.preprocessor import build_input_form_grid
from utils.config import INPUT_CATEGORICAL_FEATURES, INPUT_NUMERIC_FEATURES
from utils.ui_utils import apply_full_page_background, inject_custom_css


st.set_page_config(page_title="Maintenance Agent", page_icon="🤖", layout="wide")

apply_full_page_background()

inject_custom_css("""
div.stButton > button[kind="primary"] {
    background-color: #00ffff !important;
    padding: 1.1rem 2.6rem !important;
    font-size: 1.2rem !important;
    border: none !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    letter-spacing: 1px !important;
    width: 100% !important;
    min-height: 3.8rem !important;
}

div.stButton > button[kind="primary"] * {
    color: #000000 !important;
    font-weight: 800 !important;
}

.agent-card {
    background: rgba(8, 18, 24, 0.84);
    border: 1px solid rgba(0, 255, 255, 0.20);
    border-radius: 18px;
    padding: 1rem 1.1rem;
    margin-bottom: 1rem;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.20);
}

.agent-chip {
    display: inline-block;
    margin: 0.25rem 0.35rem 0.25rem 0;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: rgba(0, 255, 255, 0.12);
    border: 1px solid rgba(0, 255, 255, 0.35);
    color: #dffefe;
    font-size: 0.92rem;
}

.insight-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin: 1rem 0 1.2rem;
}

.insight-card {
    background: rgba(10, 16, 21, 0.82);
    border: 1px solid rgba(0, 255, 255, 0.18);
    border-radius: 16px;
    padding: 1rem;
}

.insight-label {
    color: #9ed9dd;
    font-size: 0.88rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.insight-value {
    color: #ffffff;
    font-size: 1.45rem;
    font-weight: 700;
    margin-top: 0.35rem;
}

.policy-card {
    background: rgba(18, 23, 17, 0.86);
    border-left: 4px solid #f0c419;
    border-radius: 14px;
    padding: 1rem 1rem 0.9rem;
    margin-bottom: 0.9rem;
}

.policy-card h4 {
    margin: 0 0 0.4rem 0;
}

.safety-note {
    background: rgba(61, 61, 8, 0.72);
    border: 1px solid rgba(240, 196, 25, 0.24);
    border-radius: 14px;
    padding: 0.95rem 1rem;
    margin-top: 1.1rem;
}

.trace-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin: 1rem 0 1.25rem;
}

.trace-card {
    background: linear-gradient(180deg, rgba(10, 18, 24, 0.92), rgba(8, 15, 19, 0.96));
    border: 1px solid rgba(0, 255, 255, 0.16);
    border-radius: 16px;
    padding: 1rem 1.05rem;
}

.trace-label {
    color: #9ed9dd;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.trace-value {
    color: #ffffff;
    font-size: 1.05rem;
    font-weight: 700;
    margin-top: 0.35rem;
}

.mode-pill {
    display: inline-block;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    background: rgba(0, 255, 255, 0.12);
    border: 1px solid rgba(0, 255, 255, 0.3);
    color: #dffefe;
    font-size: 0.9rem;
    margin-left: 0.5rem;
}

.why-box {
    background: rgba(10, 23, 29, 0.9);
    border: 1px solid rgba(0, 255, 255, 0.16);
    border-radius: 18px;
    padding: 1rem 1.1rem;
    margin: 1rem 0 1.25rem;
}

.why-title {
    color: #00ffff;
    font-weight: 800;
    margin-bottom: 0.45rem;
}

.why-list {
    margin: 0.35rem 0 0 1.15rem;
}

.why-list li {
    margin-bottom: 0.3rem;
}
""")


def load_agent():
    return build_agent()


def _has_meaningful_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def summarize_submission(input_data: dict, maintenance_query: str) -> dict:
    telemetry_fields = list(INPUT_NUMERIC_FEATURES) + list(INPUT_CATEGORICAL_FEATURES) + ["last_service_date"]
    filled_telemetry = [field for field in telemetry_fields if _has_meaningful_value(input_data.get(field))]

    has_query = bool(maintenance_query.strip())
    has_telemetry = bool(filled_telemetry)
    if has_query and has_telemetry:
        request_mode = "Combined"
    elif has_query:
        request_mode = "Query Only"
    elif has_telemetry:
        request_mode = "Input Only"
    else:
        request_mode = "Empty"

    return {
        "request_mode": request_mode,
        "query_present": has_query,
        "telemetry_present": has_telemetry,
        "filled_telemetry_count": len(filled_telemetry),
        "filled_telemetry_fields": filled_telemetry,
    }


def render_issue_chips(issues: list[str]) -> None:
    chips = "".join(
        f"<span class='agent-chip'>{issue.replace('_', ' ').title()}</span>"
        for issue in issues
    )
    st.markdown(chips, unsafe_allow_html=True)


def render_action_plan(action_plan: list[dict]) -> None:
    st.subheader("🛠 Agent Recommendations")
    for idx, item in enumerate(action_plan, start=1):
        st.markdown(
            f"""
            <div class="agent-card">
                <h4>{idx}. {item.get("issue", "Recommended Action")}</h4>
                <p><strong>Reason:</strong> {item.get("reason", "No reason provided.")}</p>
                <p><strong>Recommended action:</strong> {item.get("action", "No action provided.")}</p>
                <p><strong>Priority:</strong> {item.get("priority", "Unknown")} | <strong>Timeline:</strong> {item.get("timeline", "Not specified")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_policy_checks(policy_checks: list[dict]) -> None:
    if not policy_checks:
        return
    st.subheader("🏢 Fleet Policy Checks")
    for item in policy_checks:
        st.markdown(
            f"""
            <div class="policy-card">
                <h4>{item.get("title", "Policy Check")}</h4>
                <p><strong>Status:</strong> {item.get("status", "Unknown")}</p>
                <p>{item.get("detail", "No detail available.")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_decision_trace(report: dict, submission_summary: dict | None = None) -> None:
    trace = report.get("decision_trace", {}) or {}
    submission_summary = submission_summary or {}
    request_mode = str(
        submission_summary.get(
            "request_mode",
            report.get("request_mode", trace.get("request_mode", "unknown")),
        )
    ).replace("_", " ").title()
    parsed_facts = trace.get("parsed_query_facts", {}) or {}
    detected_signals = trace.get("detected_signals") or report.get("key_issues", []) or []
    if isinstance(detected_signals, list):
        detected_signals = [issue for issue in detected_signals if issue != "no immediate critical issue"]
    retrieval_query = trace.get("retrieval_query", "")
    retrieval_mode = trace.get("retrieval_mode", "KEYWORD")
    top_recommendation = trace.get("top_recommendation") or (
        report.get("action_plan", [{}])[0].get("issue", "Preventive Monitoring")
        if isinstance(report.get("action_plan"), list) and report.get("action_plan")
        else "Preventive Monitoring"
    )
    query_present = bool(submission_summary.get("query_present", trace.get("query_present")))
    telemetry_present = bool(submission_summary.get("telemetry_present", trace.get("telemetry_present")))
    telemetry_count = int(submission_summary.get("filled_telemetry_count", 0))

    st.markdown("### 🧭 Decision Trace")
    st.markdown(
        f"""
        <div class="why-box">
            <div class="why-title">How the agent formed this answer</div>
            <div>The run mode was <span class="mode-pill">{request_mode}</span>.</div>
            <ul class="why-list">
                <li>Query clues were parsed into maintenance facts before scoring: <strong>{'Yes' if query_present else 'No'}</strong>.</li>
                <li>Telemetry values were normalized and scored by the model: <strong>{'Yes' if telemetry_present else 'No'}</strong>.</li>
                <li>Filled telemetry fields: <strong>{telemetry_count}</strong>.</li>
                <li>Local retrieval mode: <strong>{retrieval_mode}</strong>.</li>
                <li>Top recommendation: <strong>{top_recommendation}</strong>.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="trace-card">
                <div class="trace-label">Query Parsed</div>
                <div class="trace-value">{'Yes' if query_present else 'No'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="trace-card">
                <div class="trace-label">Telemetry Used</div>
                <div class="trace-value">{'Yes' if telemetry_present else 'No'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="trace-card">
                <div class="trace-label">Signals Detected</div>
                <div class="trace-value">{len(detected_signals)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if parsed_facts:
        st.markdown("#### Query Facts Used")
        fact_columns = st.columns(min(3, len(parsed_facts)))
        for idx, (key, value) in enumerate(parsed_facts.items()):
            col = fact_columns[idx % len(fact_columns)]
            with col:
                st.markdown(
                    f"""
                    <div class="trace-card">
                        <div class="trace-label">{key.replace('_', ' ').title()}</div>
                        <div class="trace-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if retrieval_query:
        st.caption("Retrieval query generated from detected maintenance signals.")


def render_report(report: dict, submission_summary: dict | None = None) -> None:
    health_summary = report.get("health_summary", {})
    risk_prediction = report.get("risk_prediction", 0)
    risk_score = float(health_summary.get("risk_score", report.get("risk_score", 0.0)))
    risk_level = str(health_summary.get("risk_level", report.get("risk_level", "UNKNOWN")))
    key_issues = health_summary.get("key_issues", report.get("key_issues", []))
    maintenance_query = report.get("maintenance_query", "")
    parsed_query_facts = report.get("parsed_query_facts", {})
    policy_checks = report.get("fleet_policy_checks", [])
    submission_summary = submission_summary or {}
    request_mode = str(
        submission_summary.get(
            "request_mode",
            report.get("request_mode", "unknown"),
        )
    ).replace("_", " ").title()

    st.markdown(f"### 🚦 Agent Summary <span class='mode-pill'>{request_mode}</span>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1.2, 1, 1])
    with col1:
        if risk_prediction == 1 or risk_level.upper() == "HIGH":
            st.error("🔴 **Immediate maintenance attention required**")
        elif risk_level.upper() == "MEDIUM":
            st.warning("🟠 **Maintenance attention recommended**")
        else:
            st.success("🟢 **Vehicle appears stable for now**")
    with col2:
        st.metric("Risk", f"{risk_score:.1%}")
    with col3:
        st.metric("Risk Level", risk_level.title())

    if parsed_query_facts:
        st.caption("Parsed query facts were applied to this report.")

    render_decision_trace(report, submission_summary=submission_summary)

    st.markdown("### 🔍 Detected Issues")
    render_issue_chips(key_issues)

    render_policy_checks(policy_checks)
    render_action_plan(report.get("action_plan", []))

    st.markdown(
        f"""
        <div class="safety-note">
            <strong>Safety Notice:</strong> {report.get(
                "disclaimer",
                "This assistant provides decision support only. Confirm safety-critical actions with a certified technician."
            )}
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown("""
<h1 style='font-size: 74px; text-shadow: 0 0 10px rgba(0,255,255,0.5);'>🤖 Vehicle Maintenance Agent</h1>
""", unsafe_allow_html=True)
st.caption("Use this assistant to generate a full diagnostics report with issues, maintenance guidance, and next-step recommendations.")

with st.container():
    st.markdown(
        """
        <div class="agent-card">
            <h3>What this page does</h3>
            <p>This assistant combines your trained maintenance model, rule-based diagnostics, retrieval, and optional Ollama enrichment to generate inspection priorities, service planning, and fleet policy checks.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

_, _, features = load_artifacts()
agent = load_agent()

st.markdown("---")
maintenance_query = st.text_area(
    "Maintenance Query (Optional)",
    placeholder="Example: This SUV has overheating, vibration, and overdue service. What should be inspected first?",
    help="Ask a maintenance question to guide the agent's retrieval and recommendations.",
)
input_data = build_input_form_grid(features, num_cols=2, use_sidebar=False)

run_agent = st.button("🤖 Run Maintenance Agent", type="primary")

if run_agent:
    try:
        with st.spinner("Analyzing vehicle telemetry with the maintenance agent..."):
            input_data["maintenance_query"] = maintenance_query
            submission_summary = summarize_submission(input_data, maintenance_query)

            if (
                submission_summary.get("request_mode") == "Query Only"
                and maintenance_query.strip()
                and not is_maintenance_query_relevant(maintenance_query)
            ):
                st.warning(
                    "This query looks unrelated to vehicle maintenance. "
                    "Please enter a maintenance-specific query or provide telemetry inputs."
                )
                st.stop()

            # Defensive normalization: prevent blank date strings from crashing downstream date parsing.
            if str(input_data.get("last_service_date", "")).strip() == "":
                input_data["last_service_date"] = date.today().isoformat()

            agent_state = agent.invoke({"input_data": input_data})
            report = agent_state["result"]

        st.markdown("---")
        render_report(report, submission_summary=submission_summary)
        if input_data.get("fault_code_count_unknown"):
            st.warning(
                "Fault code count was not provided, so the agent used a conservative estimate. "
                "Provide the actual count if you have it for a more reliable report."
            )
    except Exception as exc:
        # One retry path for stale sessions that may still hold old parser behavior.
        try:
            fallback_input = dict(input_data)
            fallback_input["last_service_date"] = date.today().isoformat()
            submission_summary = summarize_submission(fallback_input, maintenance_query)
            agent_state = agent.invoke({"input_data": fallback_input})
            report = agent_state["result"]
            st.info("Recovered from an input parsing issue by applying safe defaults for optional fields.")
            st.markdown("---")
            render_report(report, submission_summary=submission_summary)
        except Exception:
            st.error("The maintenance agent could not complete this run. Please validate inputs and try again.")
            st.caption(f"Technical detail: {exc}")

