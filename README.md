# Stunting Anthropometric Leakage: A Machine Learning Framework

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dediunsera/stunting-anthropometric-leakage/blob/main/Colab_Stunting_Framework.ipynb)

This repository contains the codebase, methodology, and experimental results for analyzing **data leakage** and **measurement uncertainty** in machine learning models designed for childhood stunting prediction.

## 📌 Research Problem
Machine learning models are increasingly used to predict stunting risks in children. However, these models often suffer from a critical flaw: **Anthropometric Data Leakage**. Since the ground-truth label for stunting is mathematically derived from height, age, and sex (Height-for-Age Z-score / HAZ), feeding these exact raw anthropometric variables into a predictive model causes an artificial inflation of performance metrics (such as accuracy and AUC). 
Furthermore, real-world anthropometric measurements collected in large-scale surveys are highly prone to **measurement noise**. Small human errors in measuring a child's height can lead to "label switching" (e.g., a healthy child being misclassified as stunted, or vice versa), leading to highly unreliable public health policy decisions.

## 🔬 Methodology
To address this, our research proposes a robust computational framework:
1. **Leakage Audit:** Systematically identifying and removing direct mathematical proxies of HAZ from the feature space.
2. **Measurement Uncertainty Simulation:** Applying Monte Carlo perturbations to simulate real-world anthropometric measurement errors (e.g., ± 0.5 cm to ± 2.0 cm) and observing their impact on model degradation.
3. **Decision Framework:** Developing a risk-threshold policy framework that evaluates the tradeoff between intervention coverage and misclassification costs across different demographic subgroups.

## 📊 Key Results
- **Performance Illusion:** Models utilizing leaky anthropometric features exhibit near-perfect but entirely spurious performance (AUC > 0.98), which plummets to realistic levels when deployed on strictly non-leaky socio-demographic and health features.
- **Sensitivity to Noise:** We demonstrate that even a small measurement error margin (e.g., 1 cm) causes significant label switching, severely degrading the reliability of standard diagnostic tools.
- **Policy Tradeoff:** The proposed framework provides an optimized decision boundary that minimizes exclusion errors (missing actually stunted children) while maintaining practical resource constraints for health interventions.

## 📂 Directory Structure
- `scripts/`: Python scripts for data preparation, model training, Monte Carlo simulations, and policy evaluations.
- `results/`: Output data including calibration bins, feature importance, policy curves, and metric logs.
- `figures/`: Visualizations of the methodology flowchart, ROC curves, leakage comparators, and sensitivity heatmaps.
- `Colab_Stunting_Framework.ipynb`: The main interactive Jupyter Notebook.
- `data/`: *(Private)* Directory for storing the confidential dataset locally.

## 📈 Dataset Overview (Exploratory Data Analysis)
The private dataset used in this project is based on the **Indonesian Health Survey (SKI) 2023 for Toddlers (0-59 months)**.

- **Total Samples:** 86,364 records (toddlers)
- **Total Variables:** 171 features
- **Key Feature Categories:**
  - **Geographic & Sampling:** Province, Regency/City, Urban/Rural classification (*Klasifikasi Desa/Kelurahan*), and weighting factors (*Penimbang Populasi*).
  - **Demographics:** Gender, Birth Date, and Socioeconomic indicators of parents (Highest Education, Employment Status).
  - **Health Indicators:** Medical diagnoses such as Tuberculosis (TBC), Hepatitis, Asthma, and Health Insurance ownership.
  - **Anthropometric Measures:** Measurements used for stunting risk assessment and policy tradeoff analysis.

> **Privacy Note:** Due to the sensitive nature of the demographic and health information, the raw dataset is kept strictly confidential and is **not** included in this public repository. To run the code in Colab, please upload the appropriate dataset manually.
