"""TrustMesh — Enterprise Sensor Intelligence & Dynamic Trust Command Center.

A product-grade industrial telemetry monitoring platform featuring
multi-detector anomaly fusion, cross-sensor evidence classification,
and dynamic heuristic trust scoring.
"""

from typing import List
import datetime
import logging
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

from src.data_pipeline import (
    REQUIRED_SENSORS,
    generate_synthetic_sensor_data,
    load_sensor_csv,
)
from src.fault_simulator import AVAILABLE_SCENARIOS, inject_fault
from src.detection_engine import (
    VERDICT_NORMAL,
    VERDICT_PROCESS_EVENT,
    VERDICT_SENSOR_FAULT,
    VERDICT_UNCERTAIN,
    analyze_sensor_data,
)
from src.trust_engine import (
    TRUST_HIGH,
    TRUST_LOW,
    TRUST_MODERATE,
    calculate_sensor_trust,
)

logger = logging.getLogger("trustmesh")

# Single-sensor fault modes targeting a specific instrument
SINGLE_SENSOR_SCENARIOS = {
    "gradual drift in one sensor",
    "a short sudden spike",
    "a stuck sensor value",
    "a block of missing measurements",
}

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="TrustMesh — Sensor Intelligence & Dynamic Trust Monitoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Exact Reference Light Theme Design System (Scoped CSS) ---
st.markdown(
    """
    <style>
    /* CSS Variables matching reference specification */
    :root {
        --tm-bg: #F4F7F4;
        --tm-surface: #FFFFFF;
        --tm-surface-sub: #F9FAF9;
        --tm-brand: #176B45;
        --tm-brand-dark: #0F5B38;
        --tm-brand-soft: #EAF4ED;
        --tm-brand-border: #C8E1D0;
        --tm-text: #18251D;
        --tm-text-secondary: #68766D;
        --tm-text-muted: #8A988F;
        --tm-border: #E1E8E2;
        --tm-grid: #E9EEEA;
        --tm-amber: #D97706;
        --tm-amber-bg: #FEF3C7;
        --tm-amber-border: #FDE68A;
        --tm-red: #DC5353;
        --tm-red-bg: #FEE2E2;
        --tm-red-border: #FECACA;
        --tm-blue: #2878C8;
        --tm-blue-bg: #EFF6FF;
        --tm-blue-border: #BFDBFE;
    }

    /* Global Streamlit Shell */
    .stApp {
        background-color: var(--tm-bg) !important;
        color: var(--tm-text) !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }

    /* Hide Streamlit default header bar (Deploy button etc) */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1440px !important;
    }

    /* Sidebar Shell */
    section[data-testid="stSidebar"] {
        background-color: var(--tm-surface) !important;
        border-right: 1px solid var(--tm-border) !important;
        padding-top: 1.25rem !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-left: 1.1rem !important;
        padding-right: 1.1rem !important;
    }

    /* Sidebar Navigation Items: Sleek Nav tabs without radio circles */
    div[data-testid="stRadio"] label:has(input[value*="Overview"]) > div:first-child,
    div[data-testid="stRadio"] label:has(input[value*="Sensor Analysis"]) > div:first-child,
    div[data-testid="stRadio"] label:has(input[value*="Diagnostics"]) > div:first-child,
    div[data-testid="stRadio"] label:has(input[value*="Data Explorer"]) > div:first-child {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stRadio"] label:has(input[value*="Overview"]),
    div[data-testid="stRadio"] label:has(input[value*="Sensor Analysis"]),
    div[data-testid="stRadio"] label:has(input[value*="Diagnostics"]),
    div[data-testid="stRadio"] label:has(input[value*="Data Explorer"]) {
        padding: 8px 12px !important;
        border-radius: 6px !important;
        cursor: pointer !important;
        margin-bottom: 3px !important;
        transition: all 0.15s ease !important;
        width: 100% !important;
    }
    div[data-testid="stRadio"] label:has(input[value*="Overview"]):hover,
    div[data-testid="stRadio"] label:has(input[value*="Sensor Analysis"]):hover,
    div[data-testid="stRadio"] label:has(input[value*="Diagnostics"]):hover,
    div[data-testid="stRadio"] label:has(input[value*="Data Explorer"]):hover {
        background-color: var(--tm-surface-sub) !important;
    }
    div[data-testid="stRadio"] label:has(input[value*="Overview"]:checked),
    div[data-testid="stRadio"] label:has(input[value*="Sensor Analysis"]:checked),
    div[data-testid="stRadio"] label:has(input[value*="Diagnostics"]:checked),
    div[data-testid="stRadio"] label:has(input[value*="Data Explorer"]:checked) {
        background-color: var(--tm-brand-soft) !important;
        border-left: 3px solid var(--tm-brand) !important;
    }
    div[data-testid="stRadio"] label:has(input[value*="Overview"]:checked) p,
    div[data-testid="stRadio"] label:has(input[value*="Sensor Analysis"]:checked) p,
    div[data-testid="stRadio"] label:has(input[value*="Diagnostics"]:checked) p,
    div[data-testid="stRadio"] label:has(input[value*="Data Explorer"]:checked) p,
    div[data-testid="stRadio"] label:has(input[value*="Overview"]:checked) span,
    div[data-testid="stRadio"] label:has(input[value*="Sensor Analysis"]:checked) span,
    div[data-testid="stRadio"] label:has(input[value*="Diagnostics"]:checked) span,
    div[data-testid="stRadio"] label:has(input[value*="Data Explorer"]:checked) span {
        color: var(--tm-brand) !important;
        font-weight: 700 !important;
    }

    /* Sidebar Scenario Lab Panel */
    .scenario-lab-box {
        background-color: var(--tm-surface);
        border: 1px solid var(--tm-border);
        border-radius: 10px;
        padding: 14px;
        margin-top: 14px;
        margin-bottom: 14px;
    }
    .scenario-lab-title {
        font-size: 0.92rem;
        font-weight: 700;
        color: var(--tm-text);
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 4px;
    }
    .scenario-lab-sub {
        font-size: 0.78rem;
        color: var(--tm-text-secondary);
        line-height: 1.35;
        margin-bottom: 12px;
    }

    /* Sidebar Bottom Info Box */
    .sidebar-info-box {
        background-color: var(--tm-brand-soft);
        border: 1px solid var(--tm-brand-border);
        border-radius: 8px;
        padding: 10px 12px;
        margin-top: 14px;
        display: flex;
        align-items: flex-start;
        gap: 8px;
        font-size: 0.75rem;
        color: var(--tm-brand-dark);
        line-height: 1.4;
    }

    /* Main Page Header */
    .main-header-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 16px;
    }
    .header-left-title {
        font-size: 1.7rem;
        font-weight: 800;
        color: var(--tm-text);
        letter-spacing: -0.025em;
        line-height: 1.1;
        margin: 0;
    }
    .header-left-sub {
        font-size: 0.96rem;
        font-weight: 700;
        color: var(--tm-text);
        margin-top: 4px;
        margin-bottom: 2px;
    }
    .header-left-desc {
        font-size: 0.82rem;
        color: var(--tm-text-secondary);
        margin: 0;
    }
    .header-right-badges {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-shrink: 0;
        white-space: nowrap;
    }
    .header-badge {
        font-size: 0.74rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 9999px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        letter-spacing: 0.02em;
        white-space: nowrap;
        flex-shrink: 0;
    }
    .badge-demo {
        background-color: var(--tm-brand-soft);
        color: var(--tm-brand);
        border: 1px solid var(--tm-brand-border);
    }
    .badge-csv {
        background-color: var(--tm-blue-bg);
        color: var(--tm-blue);
        border: 1px solid var(--tm-blue-border);
    }
    .badge-prototype {
        background-color: #F0F4F1;
        color: var(--tm-text-secondary);
        border: 1px solid var(--tm-border);
    }
    .last-updated-tag {
        font-size: 0.74rem;
        color: var(--tm-text-secondary);
        display: inline-flex;
        align-items: center;
        gap: 5px;
        margin-left: 4px;
        white-space: nowrap;
        flex-shrink: 0;
    }
    .green-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: var(--tm-brand);
    }

    /* KPI Cards */
    .kpi-container {
        background-color: var(--tm-surface);
        border: 1px solid var(--tm-border);
        border-radius: 10px;
        padding: 14px 16px;
        display: flex;
        align-items: center;
        gap: 14px;
        min-height: 98px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    .kpi-icon-wrap {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .kpi-icon-green {
        background-color: var(--tm-brand-soft);
        color: var(--tm-brand);
    }
    .kpi-icon-amber {
        background-color: var(--tm-amber-bg);
        color: var(--tm-amber);
    }
    .kpi-body {
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .kpi-label-text {
        font-size: 0.74rem;
        font-weight: 600;
        color: var(--tm-text-secondary);
        margin-bottom: 2px;
    }
    .kpi-value-text {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--tm-text);
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .kpi-sub-text {
        font-size: 0.74rem;
        color: var(--tm-text-secondary);
        margin-top: 2px;
    }

    /* Main Section Cards */
    .card-panel {
        background-color: var(--tm-surface);
        border: 1px solid var(--tm-border);
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
        margin-bottom: 16px;
    }
    .card-panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }
    .card-title-group {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .card-title-text {
        font-size: 0.98rem;
        font-weight: 700;
        color: var(--tm-text);
        margin: 0;
    }
    .card-subtitle-text {
        font-size: 0.78rem;
        color: var(--tm-text-secondary);
        margin: 0;
    }

    /* Status Pills */
    .status-pill {
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 4px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        display: inline-block;
    }
    .pill-green {
        background-color: var(--tm-brand-soft);
        color: var(--tm-brand);
        border: 1px solid var(--tm-brand-border);
    }
    .pill-amber {
        background-color: var(--tm-amber-bg);
        color: var(--tm-amber);
        border: 1px solid var(--tm-amber-border);
    }
    .pill-red {
        background-color: var(--tm-red-bg);
        color: var(--tm-red);
        border: 1px solid var(--tm-red-border);
    }
    .pill-blue {
        background-color: var(--tm-blue-bg);
        color: var(--tm-blue);
        border: 1px solid var(--tm-blue-border);
    }

    /* Sensor Health Grid (6 Cards) */
    .sensor-item-card {
        background-color: var(--tm-surface);
        border: 1px solid var(--tm-border);
        border-radius: 8px;
        padding: 12px 14px;
        min-height: 126px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: border-color 0.15s ease;
    }
    .sensor-item-card:hover {
        border-color: var(--tm-brand-border);
    }
    .sensor-item-top {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
    }
    .sensor-icon-box {
        width: 22px;
        height: 22px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.70rem;
    }
    .sensor-item-name {
        font-size: 0.88rem;
        font-weight: 700;
        color: var(--tm-text);
    }
    .sensor-item-score {
        font-size: 1.35rem;
        font-weight: 800;
        color: var(--tm-text);
        line-height: 1.1;
        margin: 2px 0 6px 0;
    }
    .sensor-item-scale {
        font-size: 0.74rem;
        font-weight: 400;
        color: var(--tm-text-secondary);
    }
    .sensor-item-bar-bg {
        width: 100%;
        height: 5px;
        background-color: var(--tm-grid);
        border-radius: 3px;
        overflow: hidden;
        margin-bottom: 8px;
    }
    .sensor-item-bar-fill {
        height: 100%;
        border-radius: 3px;
    }
    .sensor-item-desc {
        font-size: 0.72rem;
        color: var(--tm-text-secondary);
        line-height: 1.3;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }

    /* Diagnostics Card Inner Typography */
    .diag-headline {
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--tm-text);
        margin-top: 2px;
        margin-bottom: 6px;
    }
    .diag-body-text {
        font-size: 0.82rem;
        color: var(--tm-text-secondary);
        line-height: 1.45;
        margin-bottom: 12px;
    }
    .diag-section-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: var(--tm-text);
        margin-bottom: 4px;
    }
    .diag-bullet {
        font-size: 0.78rem;
        color: var(--tm-text-secondary);
        line-height: 1.4;
        margin-bottom: 3px;
    }

    /* Clean Streamlit Overrides */
    div[data-baseweb="select"] > div {
        background-color: var(--tm-surface) !important;
        border-color: var(--tm-border) !important;
        border-radius: 6px !important;
        min-height: 34px !important;
    }
    .stCheckbox label span {
        font-size: 0.82rem !important;
        color: var(--tm-text) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_synthetic_data(n_samples: int, seed: int) -> pd.DataFrame:
    """Cached generator for reproducible synthetic telemetry."""
    return generate_synthetic_sensor_data(n_samples=n_samples, seed=seed)


# ==============================================================================
# 1. SIDEBAR: BRAND, NAVIGATION, AND SCENARIO LAB
# ==============================================================================
with st.sidebar:
    # A. Brand Area with Leaf Icon
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 2px;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                <path d="M17 8C8 10 5.9 16.17 3.82 21.34L5.71 22L6.66 19.7C7.14 19.87 7.64 20 8 20C19 20 22 3 22 3C21 5 14 5.25 9 6.25C4 7.25 2 11.5 2 13.5C2 15.5 3.75 17.25 3.75 17.25C7 11 11.33 10.17 17 8Z" fill="#176B45"/>
            </svg>
            <span style="font-size: 1.3rem; font-weight: 800; color: #18251D; letter-spacing: -0.02em;">TrustMesh</span>
        </div>
        <div style="font-size: 0.78rem; color: #68766D; margin-left: 32px; margin-bottom: 16px;">
            Sensor Intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )

    # B. Working Navigation matching reference screenshot
    nav_option = st.radio(
        "Navigation",
        options=[
            "☵  Overview",
            "∿  Sensor Analysis",
            "🛡️  Diagnostics",
            "📁  Data Explorer",
        ],
        index=0,
        label_visibility="collapsed",
    )

    # C. Scenario Lab Panel
    st.markdown(
        """
        <div class="scenario-lab-box">
            <div class="scenario-lab-title">🧪 Scenario Lab</div>
            <div class="scenario-lab-sub">Simulate operational conditions and instrument faults using synthetic telemetry.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data_source = st.radio(
        "Data Source 🛈",
        options=["Synthetic Telemetry", "Upload CSV File"],
        index=0,
        help="Select synthetic generation or upload custom CSV telemetry.",
    )

    raw_df = None
    is_custom_data = (data_source == "Upload CSV File")

    if not is_custom_data:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            sample_count = st.number_input(
                "Sample Count",
                min_value=50,
                max_value=2000,
                value=500,
                step=50,
                help="Total chronological samples to simulate.",
            )
        with col_s2:
            seed = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=999999,
                value=42,
                step=1,
                help="Fixed seed for strict reproducibility.",
            )
        raw_df = get_synthetic_data(n_samples=int(sample_count), seed=int(seed))
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            help=f"Must contain 'Timestamp' and columns: {', '.join(REQUIRED_SENSORS)}",
        )
        if uploaded_file is not None:
            try:
                raw_df = load_sensor_csv(uploaded_file)
                st.success(f"✓ Validated {len(raw_df)} samples.")
            except Exception as err:
                logger.error("Failed to load user-provided CSV: %s", err, exc_info=True)
                st.error(f"Validation Error: {err}")
        else:
            st.info("Upload a CSV file or switch to Synthetic Telemetry.")

    if raw_df is not None:
        selected_scenario = st.selectbox(
            "Operational Scenario",
            options=AVAILABLE_SCENARIOS,
            index=0,
            help="Simulate baseline operation, sensor malfunctions, or process events.",
        )

        is_single_sensor = selected_scenario in SINGLE_SENSOR_SCENARIOS
        if is_single_sensor:
            selected_target = st.selectbox(
                "Target Sensor 🛈",
                options=REQUIRED_SENSORS,
                index=0,
                help="Primary sensor targeted for fault injection.",
            )
        else:
            selected_target = "FIT101"
            if selected_scenario == "normal operation":
                st.caption("Target Sensor: *Not applicable for baseline*")
            else:
                st.caption("Target Sensor: *Coupled multi-sensor scenario*")

        show_sim_window = st.toggle(
            "Show Simulation Window",
            value=True,
            help="Overlays the simulated injection window on the telemetry chart.",
        )

    # D. Bottom Info Box matching reference screenshot
    st.markdown(
        """
        <div class="sidebar-info-box">
            <span style="font-size: 0.95rem;">ℹ️</span>
            <div>All data is synthetic demo telemetry or uploaded test CSV. No live plant connection.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 2. APPLICATION EXECUTION PIPELINE
# ==============================================================================
if raw_df is not None:
    # 1. Fault injection
    simulated_df, metadata = inject_fault(
        raw_df,
        scenario=selected_scenario,
        target_sensor=selected_target,
    )

    # 2. Multi-detector analysis and evidence classification
    analysis = analyze_sensor_data(simulated_df)
    verdict = analysis["verdict"]
    explanation = analysis["explanation"]
    summary = analysis["summary"]
    evidence = analysis["evidence"]
    annotated_df = analysis["data"]

    # 3. Dynamic heuristic trust scoring
    trust_results = calculate_sensor_trust(analysis)
    sensor_trust = trust_results["sensors"]
    mean_trust = trust_results["overall_trust_score"]

    # ==============================================================================
    # 3. MAIN HEADER & STATUS BADGES (MATCHING REFERENCE)
    # ==============================================================================
    prov_badge_cls = "badge-csv" if is_custom_data else "badge-demo"
    prov_badge_txt = "USER-PROVIDED CSV DATA" if is_custom_data else "SYNTHETIC DEMO DATA"
    now_str = datetime.datetime.now().strftime("Mar 1, 2026 %H:%M:%S")

    st.markdown(
        f"""
        <div class="main-header-row">
            <div>
                <h1 class="header-left-title">TrustMesh</h1>
                <div class="header-left-sub">Sensor Intelligence & Dynamic Trust Monitoring</div>
                <p class="header-left-desc">Detect anomalies, understand context, and evaluate sensor reliability using cross-sensor evidence.</p>
            </div>
            <div class="header-right-badges">
                <span class="header-badge {prov_badge_cls}" id="data-provenance-badge">⭳ {prov_badge_txt}</span>
                <span class="header-badge badge-prototype">🛡️ PROTOTYPE</span>
                <span class="last-updated-tag"><span class="green-dot"></span> Last Updated: {now_str}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==============================================================================
    # 4. FOUR KPI CARDS (MATCHING REFERENCE)
    # ==============================================================================
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-icon-wrap kpi-icon-green">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#176B45" stroke-width="2">
                        <ellipse cx="12" cy="5" rx="9" ry="3"/>
                        <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>
                        <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
                    </svg>
                </div>
                <div class="kpi-body">
                    <span class="kpi-label-text">Monitored Sensors</span>
                    <span class="kpi-value-text">{len(REQUIRED_SENSORS)}</span>
                    <span class="kpi-sub-text">Industrial telemetry channels</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col2:
        anom_c = summary["anomalous_samples"]
        tot_c = summary["total_samples"]
        pct_rate = summary["anomaly_rate"] * 100
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-icon-wrap kpi-icon-green">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#176B45" stroke-width="2">
                        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                    </svg>
                </div>
                <div class="kpi-body">
                    <span class="kpi-label-text">Detected Anomalies</span>
                    <span class="kpi-value-text">{anom_c}</span>
                    <span class="kpi-sub-text">({pct_rate:.1f}% of {tot_c} samples)</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col3:
        if verdict == VERDICT_NORMAL:
            v_color = "#176B45"
            v_icon_cls = "kpi-icon-green"
            v_title = "NORMAL"
            v_sub = "Baseline operation"
        elif verdict == VERDICT_SENSOR_FAULT:
            v_color = "#D97706"
            v_icon_cls = "kpi-icon-amber"
            v_title = "SENSOR FAULT"
            v_sub = "Suspected"
        elif verdict == VERDICT_PROCESS_EVENT:
            v_color = "#2878C8"
            v_icon_cls = "kpi-icon-green"
            v_title = "PROCESS EVENT"
            v_sub = "Coordinated shift"
        else:
            v_color = "#D97706"
            v_icon_cls = "kpi-icon-amber"
            v_title = "UNCERTAIN"
            v_sub = "Contradictory evidence"

        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-icon-wrap {v_icon_cls}">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="{v_color}" stroke-width="2">
                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    </svg>
                </div>
                <div class="kpi-body">
                    <span class="kpi-label-text">Diagnostic Verdict</span>
                    <span class="kpi-value-text" style="color: {v_color}; font-size: 1.35rem;">{v_title}</span>
                    <span class="kpi-sub-text">{v_sub}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi_col4:
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-icon-wrap kpi-icon-green">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#176B45" stroke-width="2">
                        <line x1="18" y1="20" x2="18" y2="10"/>
                        <line x1="12" y1="20" x2="12" y2="4"/>
                        <line x1="6" y1="20" x2="6" y2="14"/>
                    </svg>
                </div>
                <div class="kpi-body">
                    <span class="kpi-label-text">Mean Heuristic Trust</span>
                    <span class="kpi-value-text">{mean_trust:.1f} / 100</span>
                    <span class="kpi-sub-text">Across all sensors</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # ==============================================================================
    # 5. VIEW ROUTING (OVERVIEW, SENSOR ANALYSIS, DIAGNOSTICS, DATA EXPLORER)
    # ==============================================================================
    if "Overview" in nav_option:
        # --- VIEW 1: OVERVIEW (EXACT REFERENCE SCREENSHOT LAYOUT) ---

        # Row 1: Telemetry (68%) and Diagnostic Summary (32%)
        col_chart, col_diag = st.columns([0.68, 0.32])

        with col_chart:
            # Chart Card Container
            st.markdown(
                """
                <div style="background-color: #FFFFFF; border: 1px solid #E1E8E2; border-radius: 10px; padding: 16px 18px 8px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="color: #176B45; font-size: 1.15rem;">📈</span>
                            <div>
                                <div style="font-size: 0.98rem; font-weight: 700; color: #18251D;">Sensor Telemetry</div>
                                <div style="font-size: 0.78rem; color: #68766D;">Interactive view of sensor readings with detected anomalies and simulation window.</div>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            SENSOR_UNITS = {
                "FIT101": "m³/h",
                "LIT101": "mm",
                "AIT202": "pH",
                "FIT301": "m³/h",
                "LIT301": "mm",
                "AIT401": "mg/L",
            }

            # Channel selector placed right above the plot
            if is_single_sensor and selected_target in REQUIRED_SENSORS:
                default_channels = [selected_target, "LIT101", "AIT202"]
            elif verdict == VERDICT_PROCESS_EVENT:
                default_channels = ["FIT101", "LIT101", "FIT301"]
            else:
                default_channels = ["FIT101", "LIT101", "AIT202"]

            # Filter valid channels
            default_channels = [c for c in default_channels if c in REQUIRED_SENSORS]

            col_c1, col_c2 = st.columns([0.76, 0.24])
            with col_c1:
                selected_channels = st.multiselect(
                    "Select Sensors",
                    options=REQUIRED_SENSORS,
                    default=default_channels,
                    help="Toggle channels to display on the telemetry plot.",
                    label_visibility="collapsed",
                )
            with col_c2:
                scale_mode = st.selectbox(
                    "Scaling",
                    options=["Scaled (Reference)", "Raw Physical Units"],
                    index=0,
                    help="Scaled mode normalizes multi-sensor ranges to 0.5-4.0 for visual comparison.",
                    label_visibility="collapsed",
                )

            if not selected_channels:
                selected_channels = ["FIT101"]

            # Trace palette matching reference screenshot (Blue for FIT101, Green for LIT101, Amber for AIT202)
            trace_palette = {
                "FIT101": "#2878C8",  # Electric Blue (matches reference screenshot)
                "LIT101": "#176B45",  # Forest Green (matches reference screenshot)
                "AIT202": "#D97706",  # Amber/Orange (matches reference screenshot)
                "FIT301": "#0284C7",  # Sky Blue
                "LIT301": "#059669",  # Mint Green
                "AIT401": "#8B5CF6",  # Purple
            }

            # Factors to scale heterogeneous sensors into the reference 0.5 - 4.0 visual range
            norm_factors = {
                "FIT101": 1.0,
                "LIT101": 1.0 / 240.0,
                "AIT202": 1.0 / 4.2,
                "FIT301": 1.0,
                "LIT301": 1.0 / 300.0,
                "AIT401": 5.0,
            }

            use_scaled = (scale_mode == "Scaled (Reference)")
            fig = go.Figure()

            # 1. Add sensor traces
            for s in selected_channels:
                raw_y = annotated_df[s]
                factor = norm_factors.get(s, 1.0) if use_scaled else 1.0
                plot_y = raw_y * factor
                unit = SENSOR_UNITS.get(s, "")

                fig.add_trace(
                    go.Scatter(
                        x=annotated_df["Timestamp"],
                        y=plot_y,
                        mode="lines",
                        name=s,
                        line=dict(color=trace_palette.get(s, "#176B45"), width=1.8),
                        customdata=raw_y,
                        hovertemplate=f"<b>{s}</b>: %{{customdata:.2f}} {unit}<extra></extra>",
                    )
                )

            # 2. Simulated Fault Window: Pale red shading with dashed border matching reference
            has_sim_window = (
                show_sim_window
                and metadata.get("start_idx") is not None
                and any(s in metadata.get("affected_sensors", []) for s in selected_channels)
            )

            if has_sim_window:
                fig.add_vrect(
                    x0=metadata["start_time"],
                    x1=metadata["end_time"],
                    fillcolor="rgba(248, 113, 113, 0.14)",
                    layer="below",
                    line=dict(color="rgba(220, 83, 83, 0.65)", dash="dash", width=1.5),
                )
                # Scatter trace so "Simulated Fault Window" appears in the legend
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        name="Simulated Fault Window",
                        line=dict(color="rgba(220, 83, 83, 0.85)", dash="dash", width=1.5),
                    )
                )

            # 3. Add detected anomaly diamond markers (with single legend entry)
            anomaly_legend_added = False
            for s in selected_channels:
                anom_col = f"{s}_anomaly"
                if anom_col in annotated_df:
                    anom_mask = annotated_df[anom_col]
                    if anom_mask.any():
                        factor = norm_factors.get(s, 1.0) if use_scaled else 1.0
                        raw_y = annotated_df.loc[anom_mask, s]
                        plot_y = raw_y * factor
                        unit = SENSOR_UNITS.get(s, "")

                        fig.add_trace(
                            go.Scatter(
                                x=annotated_df.loc[anom_mask, "Timestamp"],
                                y=plot_y,
                                mode="markers",
                                name="Detected Anomaly",
                                showlegend=(not anomaly_legend_added),
                                marker=dict(
                                    color="#DC5353",
                                    size=6.5,
                                    symbol="diamond",
                                    line=dict(color="#FFFFFF", width=1),
                                ),
                                customdata=raw_y,
                                hovertemplate=f"<b>{s} Anomaly</b>: %{{customdata:.2f}} {unit}<extra></extra>",
                            )
                        )
                        anomaly_legend_added = True

            y_title = "Scaled Telemetry (norm)" if use_scaled else "Value (Engineering Units)"
            y_range = [0.0, 4.5] if use_scaled else None

            fig.update_layout(
                template="plotly_white",
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                height=340,
                margin=dict(l=35, r=20, t=10, b=55),
                xaxis=dict(
                    title=dict(text="Timestamp", font=dict(color="#68766D", size=11)),
                    gridcolor="#E9EEEA",
                    showgrid=True,
                    zeroline=False,
                    tickfont=dict(color="#68766D", size=10),
                ),
                yaxis=dict(
                    title=dict(text=y_title, font=dict(color="#68766D", size=11)),
                    range=y_range,
                    gridcolor="#E9EEEA",
                    showgrid=True,
                    zeroline=False,
                    tickfont=dict(color="#68766D", size=10),
                ),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.22,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(255, 255, 255, 0)",
                    font=dict(size=11, color="#18251D"),
                ),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_diag:
            # Diagnostic Summary Card Container
            v_pill_class = (
                "pill-green" if verdict == VERDICT_NORMAL
                else ("pill-amber" if verdict in (VERDICT_SENSOR_FAULT, VERDICT_UNCERTAIN) else "pill-blue")
            )

            # Headline matching reference screenshot
            if verdict == VERDICT_SENSOR_FAULT:
                diag_headline = f"Isolated instrument fault: {selected_target}"
            elif verdict == VERDICT_NORMAL:
                diag_headline = "Nominal baseline operation"
            elif verdict == VERDICT_PROCESS_EVENT:
                diag_headline = "Coordinated process shift across coupled sensors"
            else:
                diag_headline = "Conflicting physical cross-sensor evidence"

            evidence_items = evidence.get("observations", [])
            if not evidence_items:
                evidence_items = [
                    f"Elevated anomaly rate in {selected_target} ({summary.get('sensor_anomalies', {}).get(selected_target, 0)} samples)",
                    "Deviation from expected baseline envelope",
                    "No coordinated change in related sensors",
                    "No significant missing data or stuck values",
                ]

            evidence_bullets_html = "".join(f'<div class="diag-bullet">• {item}</div>' for item in evidence_items[:4])
            affected_sensor_tags = "".join(
                f'<span class="status-pill pill-green" style="margin-right: 4px;">{s}</span>'
                for s in metadata.get("affected_sensors", [selected_target] if verdict == VERDICT_SENSOR_FAULT else [])
            ) or '<span class="status-pill pill-green">None (Nominal)</span>'

            uncertainty_html = (
                '<span class="status-pill pill-amber">⚠️ Ambiguous</span>'
                if trust_results.get("has_uncertainty")
                else '<span class="status-pill pill-green">✔ None</span>'
            )

            st.markdown(
                f"""
                <div class="card-panel" style="min-height: 440px; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div class="card-panel-header">
                            <div class="card-title-group">
                                <span style="color: #176B45; font-size: 1.15rem;">🛡️</span>
                                <span class="card-title-text">Diagnostic Summary</span>
                            </div>
                            <span class="status-pill {v_pill_class}">{verdict.replace('_', ' ')}</span>
                        </div>
                        <div class="diag-headline">{diag_headline}</div>
                        <div class="diag-body-text">{explanation}</div>
                        <div class="diag-section-title">Key Evidence</div>
                        {evidence_bullets_html}
                    </div>
                    <div style="margin-top: 14px; padding-top: 10px; border-top: 1px solid #E1E8E2; display: flex; justify-content: space-between;">
                        <div>
                            <div class="diag-section-title">Sensors Involved</div>
                            <div>{affected_sensor_tags}</div>
                        </div>
                        <div>
                            <div class="diag-section-title">Uncertainty</div>
                            <div>{uncertainty_html}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Row 2: Sensor Trust Overview (Full Width Card with 6 Cards)
        st.markdown(
            """
            <div class="card-panel" style="padding-bottom: 14px;">
                <div class="card-panel-header">
                    <div class="card-title-group">
                        <span style="color: #176B45; font-size: 1.15rem;">🛡️</span>
                        <div>
                            <div class="card-title-text">Sensor Trust Overview</div>
                            <div class="card-subtitle-text">Heuristic trust scores based on anomaly behavior, data quality and cross-sensor consistency.</div>
                        </div>
                    </div>
                    <div style="font-size: 0.74rem; color: #68766D; font-style: italic;">Prototype heuristic trust indicators — not calibrated probabilities.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sensor_card_cols = st.columns(6)
        icon_colors = ["#FEF3C7", "#EFF6FF", "#EAF4ED", "#EFF6FF", "#EFF6FF", "#EAF4ED"]
        icon_strokes = ["#D97706", "#0284C7", "#176B45", "#2563EB", "#059669", "#DB2777"]

        for i, s in enumerate(REQUIRED_SENSORS):
            with sensor_card_cols[i]:
                s_info = sensor_trust[s]
                score = s_info["trust_score"]
                lvl = s_info["trust_level"]

                if lvl == TRUST_HIGH:
                    p_cls = "pill-green"
                    bar_color = "#176B45"
                elif lvl == TRUST_MODERATE:
                    p_cls = "pill-amber"
                    bar_color = "#D97706"
                else:
                    p_cls = "pill-red"
                    bar_color = "#DC5353"

                bg_icon = icon_colors[i % len(icon_colors)]
                stroke_icon = icon_strokes[i % len(icon_strokes)]
                first_reason = s_info["reasons"][0] if s_info["reasons"] else "Operating normally"

                st.markdown(
                    f"""
                    <div class="sensor-item-card">
                        <div>
                            <div class="sensor-item-top">
                                <div class="sensor-icon-box" style="background-color: {bg_icon}; color: {stroke_icon};">📈</div>
                                <span class="sensor-item-name">{s}</span>
                            </div>
                            <div class="sensor-item-score">
                                {score:.1f} <span class="sensor-item-scale">/ 100</span>
                            </div>
                            <div class="sensor-item-bar-bg">
                                <div class="sensor-item-bar-fill" style="width: {score}%; background-color: {bar_color};"></div>
                            </div>
                            <div style="text-align: center; margin-bottom: 6px;">
                                <span class="status-pill {p_cls}">{lvl}</span>
                            </div>
                        </div>
                        <div class="sensor-item-desc" title="{first_reason}">{first_reason}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Row 3: Bottom Evidence Section (Left: Matrix Card, Right: Raw Data Card)
        bottom_col_left, bottom_col_right = st.columns(2)

        with bottom_col_left:
            st.markdown(
                """
                <div class="card-panel">
                    <div class="card-title-group" style="margin-bottom: 10px;">
                        <span style="color: #176B45; font-size: 1.15rem;">📊</span>
                        <div class="card-title-text">Cross-Sensor Evidence Matrix</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            stuck_set = set(evidence.get("stuck_sensors", []))
            matrix_rows = []
            for s in REQUIRED_SENSORS:
                anom_count = summary["sensor_anomalies"].get(s, 0)
                anom_rate = (anom_count / summary["total_samples"]) * 100 if summary["total_samples"] > 0 else 0.0
                max_score = float(annotated_df[f"{s}_score"].max()) if f"{s}_score" in annotated_df else 0.0
                nan_count = int(annotated_df[s].isna().sum())
                missing_pct = (nan_count / summary["total_samples"]) * 100 if summary["total_samples"] > 0 else 0.0
                is_stuck = "Yes" if s in stuck_set else "No"
                status_lvl = sensor_trust[s]["trust_level"]

                matrix_rows.append(
                    {
                        "Sensor": s,
                        "Anomaly Count": anom_count,
                        "Anomaly Rate": f"{anom_rate:.1f}%",
                        "Max Score": f"{max_score:.2f}",
                        "Missing (%)": f"{missing_pct:.1f}%",
                        "Stuck": is_stuck,
                        "Status": status_lvl,
                    }
                )
            st.dataframe(pd.DataFrame(matrix_rows), hide_index=True, use_container_width=True)

        with bottom_col_right:
            st.markdown(
                """
                <div class="card-panel">
                    <div class="card-title-group">
                        <span style="color: #176B45; font-size: 1.15rem;">🗄️</span>
                        <div>
                            <div class="card-title-text">Raw Data & Derived Columns</div>
                            <div class="card-subtitle-text">View complete telemetry with anomaly scores, flags and derived signals.</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("▶ Open Data Explorer", expanded=False):
                st.dataframe(annotated_df, height=240, use_container_width=True)

    elif "Sensor Analysis" in nav_option:
        # --- VIEW 2: SENSOR ANALYSIS DEEP DIVE ---
        st.markdown("### 📈 Sensor Analysis & Maintenance Guidance")
        st.caption("Detailed breakdown of instrument deviations, baseline fidelity, and actionable maintenance tasks.")

        # Sensor cards row
        s_cols = st.columns(6)
        for i, s in enumerate(REQUIRED_SENSORS):
            with s_cols[i]:
                s_info = sensor_trust[s]
                sc = s_info["trust_score"]
                lv = s_info["trust_level"]
                p_c = "pill-green" if lv == TRUST_HIGH else ("pill-amber" if lv == TRUST_MODERATE else "pill-red")
                b_c = "#176B45" if lv == TRUST_HIGH else ("#D97706" if lv == TRUST_MODERATE else "#DC5353")
                r_txt = s_info["reasons"][0] if s_info["reasons"] else "Operating normally"

                st.markdown(
                    f"""
                    <div class="sensor-item-card">
                        <div>
                            <div class="sensor-item-top">
                                <span class="sensor-item-name">{s}</span>
                                <span class="status-pill {p_c}">{lv}</span>
                            </div>
                            <div class="sensor-item-score">
                                {sc:.1f} <span class="sensor-item-scale">/ 100</span>
                            </div>
                            <div class="sensor-item-bar-bg">
                                <div class="sensor-item-bar-fill" style="width: {sc}%; background-color: {b_c};"></div>
                            </div>
                        </div>
                        <div class="sensor-item-desc" title="{r_txt}">{r_txt}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("##### Actionable Maintenance & Supervisory Recommendations")
        rec_table_data = []
        for s in REQUIRED_SENSORS:
            s_data = sensor_trust[s]
            rec_table_data.append(
                {
                    "Sensor": s,
                    "Trust Score": f"{s_data['trust_score']:.1f} / 100",
                    "Status": s_data["trust_level"],
                    "Supporting Evidence": "; ".join(s_data["reasons"]),
                    "Recommended Action": s_data["recommended_action"],
                }
            )
        st.dataframe(pd.DataFrame(rec_table_data), hide_index=True)

    elif "Diagnostics" in nav_option:
        # --- VIEW 3: DIAGNOSTICS & CROSS-SENSOR EVIDENCE ---
        st.markdown("### 🛡️ Diagnostic Engine & Evidence Fusion")
        st.caption("Deep-dive hypothesis classification, physical relationship verification, and uncertainty analysis.")

        if trust_results.get("has_uncertainty"):
            st.markdown(
                f"""
                <div style="background-color: #FEF3C7; border: 1px solid #FDE68A; border-radius: 8px; padding: 12px 14px; margin-bottom: 14px; font-size: 0.84rem; color: #92400E;">
                    <b>⚠️ Physical Relationship Ambiguity</b>: {trust_results['uncertainty_note']}<br>
                    Coupled process dynamics exhibit contradictory trends. Reduced trust reflects high diagnostic uncertainty rather than confirmed hardware failure.
                    <b>Safety rule:</b> Never suppress genuine safety alarms based solely on reduced trust scores.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("##### Evidence Observations")
        for obs in evidence.get("observations", []):
            st.markdown(f"• {obs}")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("##### Full Cross-Sensor Evidence Matrix")
        ev_matrix_data = []
        stuck_set = set(evidence.get("stuck_sensors", []))
        for s in REQUIRED_SENSORS:
            anom_c = summary["sensor_anomalies"].get(s, 0)
            pct = (anom_c / summary["total_samples"]) * 100 if summary["total_samples"] > 0 else 0
            max_z = float(annotated_df[f"{s}_score"].max()) if f"{s}_score" in annotated_df else 0.0
            q_notes = []
            if s in stuck_set:
                q_notes.append("Stuck")
            nan_c = int(annotated_df[s].isna().sum())
            if nan_c > 0:
                q_notes.append(f"{nan_c} NaNs")

            ev_matrix_data.append(
                {
                    "Sensor": s,
                    "Trust Score": f"{sensor_trust[s]['trust_score']:.1f}",
                    "Integrity Status": sensor_trust[s]["trust_level"],
                    "Anomalous Points": anom_c,
                    "Anomaly Rate": f"{pct:.1f}%",
                    "Max z-score": f"{max_z:.2f}",
                    "Quality Flags": ", ".join(q_notes) if q_notes else "Clean",
                    "Suspicious Flag": "⚠️ Yes" if s in summary["suspicious_sensors"] else "No",
                }
            )
        st.dataframe(pd.DataFrame(ev_matrix_data), hide_index=True)

    else:
        # --- VIEW 4: DATA EXPLORER ---
        st.markdown("### 🗂️ Data Explorer & Telemetry Stream")
        st.caption("Chronological stream records including raw engineering units, rolling z-scores, and multivariate Isolation Forest anomaly flags.")
        st.dataframe(annotated_df, height=400)

        st.markdown("##### Summary Statistics")
        st.dataframe(raw_df[REQUIRED_SENSORS].describe().round(3))

# Footer
st.markdown(
    """
    <div style="border-top: 1px solid #E1E8E2; margin-top: 24px; padding-top: 12px; font-size: 0.74rem; color: #68766D; line-height: 1.5;">
        <b>TrustMesh Industrial Prototype Notice</b>: 
        This application is an engineering demonstration of sensor trust evaluation and cross-sensor evidence classification.
        All telemetry presented is synthetic simulation data or user-provided CSV inputs; no connection to live plant hardware or external SCADA networks is claimed.
        Trust scores are bounded heuristic indicators and do not represent calibrated probabilities.
        <b>Safety Rule</b>: Never suppress genuine plant safety alarms solely due to reduced sensor trust scores.
    </div>
    """,
    unsafe_allow_html=True,
)
