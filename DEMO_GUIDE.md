# TrustMesh — 3-Minute Hackathon Demo Script & Showcase Guide

This guide provides a structured, repeatable 3-minute presentation workflow for pitching **TrustMesh** to hackathon judges, operators, or engineering leads.

> **Notice & Disclaimers**: 
> - TrustMesh is a **software development prototype**, not a certified or deployed industrial safety system.
> - All telemetry shown is **synthetic industrial demo data**. No connection to live physical equipment or external benchmark datasets (such as SWaT) is claimed.
> - Trust scores (0–100) are **transparent heuristic indicators based on empirical evidence**, not calibrated probabilities.
> - **Core Safety Rule**: Never suppress or ignore a genuine plant safety alarm solely because a trust score is reduced.

---

## Pitch Narrative Outline (Total Duration: ~3 Minutes)

| Timing | Segment | Focus | Dashboard Action |
| :--- | :--- | :--- | :--- |
| **0:00 – 0:30** | **Opening Hook** | Industrial Sensor Problem: Drift, false alarms, contradictory signals | Introduce header & architecture overview |
| **0:30 – 1:00** | **Step 1: Normal Baseline** | Baseline telemetry, 100% trust fidelity, multi-sensor coupling | Keep `normal operation` selected |
| **1:00 – 1:45** | **Step 2: Silent Sensor Drift** | Subtle calibration drift, localized penalty, clear recommendations | Select `gradual drift in one sensor` (`FIT101`) |
| **1:45 – 2:15** | **Step 3: Coordinated Process Event**| Multi-sensor load change vs sensor failure (avoiding false alarms) | Select `a coordinated process change` |
| **2:15 – 2:45** | **Step 4: Conflicting Evidence** | Contradictory physics, uncertainty alert, safety safeguards | Select `a conflicting-evidence scenario` |
| **2:45 – 3:00** | **Closing Summary** | Value: Trust-calibrated operations, downtime prevention, explainability | Summarize KPIs & Evidence Matrix |

---

## Detailed Step-by-Step Presentation Script

### 1. Opening Hook: The Problem (0:00 – 0:30)
- **Speaker**: 
  > *"Modern industrial plants rely on hundreds of sensors. But when an instrument drifts or produces anomalous readings, operators face a dilemma: Is the sensor broken, or is the physical process failing? Traditional rule-based thresholds produce false alarms or miss silent calibration drift. TrustMesh combines unsupervised anomaly detection with multi-sensor physical correlation to produce dynamic, explainable sensor trust scores."*
- **Visual**: Point to the top banner and KPI row showing the 6 monitored channels (`FIT101`, `LIT101`, `AIT202`, `FIT301`, `LIT301`, `AIT401`).

---

### 2. Step 1: Normal Baseline Operation (0:30 – 1:00)
- **Dashboard Action**: 
  - Ensure **Operational Scenario** is set to `normal operation`.
  - In the telemetry channel selector, select `FIT101` and `LIT101`.
- **Expected Behavior**:
  - Diagnostic Verdict: `NORMAL OPERATION` (Emerald green banner).
  - KPI Row: 0 detected anomalies, Mean Trust Index is **100.0 / 100** (`🟢 STABLE`).
  - Trust Hierarchy Chart: All 6 sensor bars are green at 100.0 (`HIGH`).
  - Plotly Telemetry: Clean cyclical industrial oscillations reflecting nominal pump operations and tank levels.
- **Speaker**:
  > *"Under normal operating conditions, all sensors operate within expected rolling baselines. TrustMesh establishes dynamic residual envelopes rather than static bounds, maintaining high baseline trust without nuisance trips."*

---

### 3. Step 2: Silent Sensor Drift (1:00 – 1:45)
- **Dashboard Action**:
  - In the sidebar **Operational Scenario**, select `gradual drift in one sensor`.
  - Ensure **Target Sensor** is `FIT101`.
  - Check **Highlight Injected Scenario Window on Chart**.
- **Expected Behavior**:
  - Diagnostic Verdict: Switches to `SENSOR FAULT SUSPECTED` (Red/Amber banner).
  - Telemetry Chart: `FIT101` shows red diamond anomaly markers during the drift window, while unperturbed channels (`LIT101`, etc.) stay clean.
  - Trust Cards & Bar Chart: `FIT101` trust drops from 100 down to **MODERATE** (~72–75), while untouched sensors remain **HIGH** (100.0).
  - Recommendation Table: Specifically identifies `FIT101` for *"routine calibration and zero-check"*, noting uncorroborated deviation from baseline.
- **Speaker**:
  > *"Here, FIT101 experiences a silent linear calibration drift. A single static threshold would either miss this or fire too late. TrustMesh isolates the anomaly strictly to FIT101, lowers its trust score to Moderate, while preserving High trust on all other sensors. It provides an immediate actionable recommendation to recalibrate."*

---

### 4. Step 3: Coordinated Process Event vs. False Alarm (1:45 – 2:15)
- **Dashboard Action**:
  - In the sidebar **Operational Scenario**, select `a coordinated process change affecting multiple related sensors`.
  - Select `FIT101`, `LIT101`, and `FIT301` in the channel multiselect.
- **Expected Behavior**:
  - Diagnostic Verdict: Changes to `PROCESS EVENT SUSPECTED` (Blue banner).
  - Explanation: Explicitly states that simultaneous shifts occurred across coupled sensors (`FIT101`, `LIT101`, `FIT301`), suggesting a plant operating transition rather than sensor failure.
  - Trust Hierarchy: Crucially, participating sensors are **not penalized as broken hardware**; trust remains **HIGH** ($\ge 80$).
- **Speaker**:
  > *"Now consider a critical scenario: what if a valve opens or demand surges? FIT101, LIT101, and FIT301 all spike simultaneously. Naive single-sensor detectors would scream that all three sensors failed! TrustMesh detects that the sensors moved in physical coordination and correctly classifies this as a Process Event, preserving High trust on all instruments."*

---

### 5. Step 4: Conflicting Evidence & Uncertainty (2:15 – 2:45)
- **Dashboard Action**:
  - In the sidebar **Operational Scenario**, select `a conflicting-evidence scenario for later uncertainty testing`.
  - Target Sensor: `FIT101`.
- **Expected Behavior**:
  - Diagnostic Verdict: Changes to `UNCERTAIN DIAGNOSIS` (Yellow/Violet banner).
  - Prominent Warning Banner: `⚠️ Unresolved Cross-Sensor Uncertainty` alert appears.
  - Observation: Notes contradictory physical coupling between inflow (`FIT101`) and tank level (`LIT101`).
  - Safety Reminder: Explicitly reminds operators: *"Never suppress genuine safety alarms based solely on reduced trust scores."*
- **Speaker**:
  > *"In edge cases where physics conflict—for instance, inflow surges while tank level paradoxically drops—TrustMesh does not force a guess. It flags the condition as Uncertain, alerts the operator to investigate conflicting cross-sensor physics, and enforces our core safety directive: never suppress a safety alarm during operational ambiguity."*

---

### 6. Closing Summary (2:45 – 3:00)
- **Speaker**:
  > *"TrustMesh provides transparency, reduces nuisance false alarms, and detects subtle calibration degradation before it causes unplanned outages. It turns raw SCADA telemetry into trustworthy, explainable intelligence."*
- **Visual**: Scroll briefly to the **Cross-Sensor Empirical Evidence Matrix** and the collapsed **Raw Telemetry Stream** expander to show complete data transparency.
