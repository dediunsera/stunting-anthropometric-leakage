"""Build Colab_Stunting_Framework.ipynb from the pipeline scripts (one section per stage)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("# Reliable and Survey-Aware ML for Childhood Stunting Screening under Anthropometric Measurement Uncertainty\n"
   "Cross-check notebook (Google Colab). Run cells top to bottom.\n\n"
   "**Before running:** upload to Google Drive folder `MyDrive/stunting_framework/data/` the files\n"
   "`SKI 2023 Balita 0-59 Bulan_asli labels.xlsx` and `lenanthro.txt`.\n\n"
   "Expected runtime on Colab CPU: ~40-60 min (Random Forest CV is the slowest step). "
   "Numbers should match `results/*.csv` in the project folder (seed = 42); tiny differences "
   "(<0.002 AUROC) can occur with other scikit-learn versions.")
code("from google.colab import drive\ndrive.mount('/content/drive')\n"
     "import os\nPROJ = '/content/drive/MyDrive/stunting_framework'\n"
     "os.makedirs(PROJ + '/scripts', exist_ok=True); os.makedirs(PROJ + '/data', exist_ok=True)\n"
     "os.environ['SKI_PROJECT_DIR'] = PROJ\n"
     "!pip -q install scikit-learn==1.8.0 openpyxl joblib\n"
     "import sklearn, pandas, numpy; print(sklearn.__version__, pandas.__version__, numpy.__version__)")
for fn, title in [("plot_style.py", "Plot style helper"), ("common.py", "Shared configuration & metric functions")]:
    md(f"## {title} (`{fn}`)")
    code(f"%%writefile {{PROJ}}/scripts/{fn}\n" + open(os.path.join(HERE, fn)).read())
code("import sys; sys.path.insert(0, PROJ + '/scripts'); os.chdir(PROJ + '/scripts')")
for fn, title in [("01_prepare_data.py", "Stage 1 - ingestion, HAZ (WHO LMS), leakage audit"),
                  ("02_ml_screening.py", "Stage 2 - survey-aware ML risk model & calibration (Axis B)"),
                  ("03_measurement_uncertainty.py", "Stage 3 - measurement uncertainty & re-measurement policies (Axis A)"),
                  ("04_decision_framework.py", "Stage 4 - derivation and evaluation of the five categories"),
                  ("05_flowcharts.py", "Flowcharts")]:
    md(f"## {title}\n`{fn}`")
    code(f"%%writefile {{PROJ}}/scripts/{fn}\n" + open(os.path.join(HERE, fn)).read())
    code(f"!cd \"{{PROJ}}/scripts\" && SKI_PROJECT_DIR=\"{{PROJ}}\" python {fn}")
md("## Show key tables and figures")
code("import pandas as pd, glob\nfrom IPython.display import Image, display\n"
     "for t in ['T_model_comparison_test_CI','T_policy_head_to_head','T_category_profile','T_draft_vs_proposed_coverage']:\n"
     "    print(t); display(pd.read_csv(f'{PROJ}/results/{t}.csv'))\n"
     "for f in sorted(glob.glob(PROJ + '/figures/*.png')):\n    print(f); display(Image(f, width=800))")
nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                    "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "..", "Colab_Stunting_Framework.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out)
