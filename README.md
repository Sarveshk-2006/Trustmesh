# TrustMesh — Industrial Sensor Intelligence & Trust Command Center

AI-based Sensor Trust, Drift & False-Alarm Detection prototype for mission-critical industrial telemetry and SCADA environments.

> **Prototype Disclaimer**: This application is a hackathon development prototype for automated sensor trust evaluation. All generated sensor telemetry is **synthetic demo data**. No connection to live plant hardware or external benchmark datasets (such as SWaT) is claimed. Sensor trust scores are **transparent heuristic indicators (0–100), not calibrated probabilities**. Never suppress a genuine safety alarm solely because a sensor has a low trust score.

---

## Project Architecture

```text
trustmesh/
├── app.py                      # Interactive Streamlit command center dashboard
├── requirements.txt            # Project dependencies
├── README.md                   # Setup, execution, and troubleshooting guide
├── DEMO_GUIDE.md               # 3-minute repeatable presentation walkthrough
├── .streamlit/
│   └── config.toml             # Industrial dark theme (charcoal & emerald styling)
├── src/
│   ├── __init__.py             # Package marker
│   ├── data_pipeline.py        # Correlated synthetic telemetry generator & schema validator
│   ├── fault_simulator.py      # Deterministic fault & operational scenario simulator
│   ├── detection_engine.py     # Rolling residual baseline & Isolation Forest anomaly fusion
│   └── trust_engine.py         # Dynamic heuristic trust scoring & actionable recommendations
└── tests/
    ├── __init__.py             # Tests package marker
    ├── test_smoke.py           # Dependency importability verification
    ├── test_data_pipeline.py   # Schema, generation, and fault simulator unit tests
    ├── test_detection_engine.py# Anomaly detection & multi-sensor evidence classification tests
    ├── test_trust_engine.py    # Dynamic trust scoring, bounded penalty, & safety rule tests
    └── test_end_to_end.py      # Comprehensive end-to-end scenario & robustness validation
```

---

## Prerequisites

- **Operating System**: Windows (tested on PowerShell), Linux, or macOS.
- **Python**: Python 3.12 (or Python 3.11 compatible).
- **Virtual Environment**: `.venv` directory configured in project root.

---

## Step-by-Step Startup Guide (Windows PowerShell)

Open **Windows PowerShell** and execute the following sequential steps:

### 1. Open the Project Directory

```powershell
Set-Location -Path "D:\trustmesh"
```

### 2. Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

*(Note: If you encounter an execution policy restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activation).*

### 3. Verify Dependencies (Optional)

If running in a fresh clone or updated environment:

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the Full Test Suite

Verify all 46 unit, integration, and robustness tests pass:

```powershell
python -m unittest discover -s tests -v
```

Expected output: `Ran 46 tests in ~15-20s ... OK`.

### 5. Start the Streamlit Application

Launch the command center dashboard (recommended port 8502 or default 8501):

```powershell
streamlit run app.py --server.port=8502
```

### 6. Open in Browser

The terminal will display the local network URL. Open your preferred browser and navigate to:

```text
http://localhost:8502
```

### 7. Stop the Application Safely

To shut down the Streamlit server, press **`Ctrl + C`** in the PowerShell terminal window.

---

## Troubleshooting

### Issue 1: Port Already in Use (e.g., Port 8502 Occupied)
If another application or previous Streamlit process is using port 8502:
1. **Option A (Choose another port)**: Run with an alternative port:
   ```powershell
   streamlit run app.py --server.port=8503
   ```
2. **Option B (Inspect and release the port)**:
   ```powershell
   Get-NetTCPConnection -LocalPort 8502 -ErrorAction SilentlyContinue
   ```
   If an old python process is holding the port, terminate it using its PID:
   ```powershell
   Stop-Process -Id <PID> -Force
   ```

### Issue 2: Script Execution Restricted by PowerShell
If `Activate.ps1` produces a script execution error (`PSSecurityException`):
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Issue 3: Missing Package or Import Error
Ensure the virtual environment python interpreter is active:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Repeatable Showcase & Presentation

To present TrustMesh in a 3-minute hackathon pitch, follow the step-by-step walkthrough in [DEMO_GUIDE.md](file:///d:/trustmesh/DEMO_GUIDE.md).
