import streamlit as st

from utils.langgraph_agent import build_agent
from utils.model_loader import load_artifacts
from utils.preprocessor import build_input_form_grid
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
""")


@st.cache_resource
def load_agent():
    return build_agent()


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


def render_report(report: dict) -> None:
    health_summary = report.get("health_summary", {})
    risk_prediction = report.get("risk_prediction", 0)
    risk_score = float(health_summary.get("risk_score", report.get("risk_score", 0.0)))
    risk_level = str(health_summary.get("risk_level", report.get("risk_level", "UNKNOWN")))
    key_issues = health_summary.get("key_issues", report.get("key_issues", []))
    maintenance_query = report.get("maintenance_query", "")
    policy_checks = report.get("fleet_policy_checks", [])

    st.markdown("### 🚦 Agent Summary")
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

    if maintenance_query:
        st.markdown("### 💬 Maintenance Query")
        st.info(maintenance_query)

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
    with st.spinner("Analyzing vehicle telemetry with the maintenance agent..."):
        input_data["maintenance_query"] = maintenance_query
        agent_state = agent.invoke({"input_data": input_data})
        report = agent_state["result"]

    st.markdown("---")
    render_report(report)
