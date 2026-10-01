"""Run the full pipeline in order (python run_all.py)."""
import subprocess, sys, time
for s in ["01_prepare_data.py", "02_ml_screening.py", "03_measurement_uncertainty.py",
          "04_decision_framework.py", "05_flowcharts.py", "06_summary_figures.py"]:
    t = time.time(); print(">>", s, flush=True)
    subprocess.run([sys.executable, s], check=True)
    print(f"   finished in {time.time()-t:.0f}s", flush=True)
