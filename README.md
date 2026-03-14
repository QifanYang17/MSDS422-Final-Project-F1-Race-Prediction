<p align="center">
  <h1 align="center">🏎️ Formula One Race Outcome Prediction</h1>
  <p align="center">
    <em>Predicting Top-10 Finishes Using 74 Years of F1 History & Machine Learning</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">
    <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
    <img src="https://img.shields.io/badge/CRISP--DM-Framework-4CAF50?style=for-the-badge" alt="CRISP-DM">
  </p>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Highlights](#-key-highlights)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Feature Engineering](#-feature-engineering)
- [Models & Results](#-models--results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Key Findings](#-key-findings)
- [References](#-references)
- [Team](#-team)

---

## 🔍 Overview

**Course:** MSDS 422 – Practical Machine Learning | Northwestern University

> *To what extent can historical Formula One race data be used to predict whether a driver will finish within the top 10 positions of a race using machine learning models that integrate driver performance history, constructor strength, circuit characteristics, and broader competitive and temporal race context?*

This project applies end-to-end machine learning techniques to **26,700+ driver–race observations** spanning the entire history of Formula One (1950–2024) to predict top-10 race finishes — the threshold for earning championship points under modern regulations. The analysis follows the **CRISP-DM** framework with strict emphasis on **data leakage prevention** and **temporal validation** to ensure realistic, deployment-ready predictions.

---

## ✨ Key Highlights

| | Highlight |
|---|---|
| 📊 | **26,700+ observations** across 1,125 races, 861 drivers, 211 constructors, and 77 circuits |
| 🔒 | **Zero data leakage** — all features use only pre-race information with `shift(1)` safeguards |
| ⏳ | **Temporal train/val/test split** (pre-2018 / 2018–2021 / 2022+) instead of random splits |
| 🤖 | **5 individual models + 4 ensemble methods** compared systematically |
| 📈 | **Best test ROC AUC: 0.869** achieved by Ensemble-Stacking |
| 🧠 | **SHAP + Permutation Importance** for full model interpretability |
| 🚀 | **FastAPI deployment template** included for production inference |

---

## 📁 Dataset

**Source:** [Formula 1 World Championship (1950–2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) — Kaggle (Vopani, 2025)

The dataset consists of **14 relational CSV tables** that are merged into a single driver–race level analytical dataset:

| Table | Description |
|---|---|
| `races.csv` | Race metadata (name, date, circuit, round) |
| `results.csv` | Race results (position, points, status, grid) |
| `qualifying.csv` | Qualifying session results |
| `drivers.csv` | Driver profiles (name, nationality, DOB) |
| `constructors.csv` | Constructor/team information |
| `driver_standings.csv` | Championship standings after each race |
| `constructor_standings.csv` | Constructor championship standings |
| `constructor_results.csv` | Constructor-level race results |
| `circuits.csv` | Circuit details (location, country, coordinates) |
| `lap_times.csv` | Individual lap timing data |
| `pit_stops.csv` | Pit stop records (timing, duration) |
| `sprint_results.csv` | Sprint race results |
| `seasons.csv` | Season-level information |
| `status.csv` | Race finish status codes (finished, DNF reasons) |

### 🎯 Target Variable Distribution

| Target | Positive Rate | Count |
|---|---|---|
| **Top-10 Finish** (primary) | 42.3% | 11,321 |
| Points Finish | 30.5% | 8,170 |
| Podium Finish | 12.7% | 3,397 |
| Did Not Finish (DNF) | 43.3% | 11,598 |

---

## 🔬 Methodology

The project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework:

<p align="center">
  <img src="notebook/crisp_dm.png" alt="CRISP-DM Framework" width="300">
</p>

### 1. Business & Data Understanding
- Defined top-10 finish prediction as a **binary classification** problem
- Explored 74 years of F1 data across multiple regulatory eras
- Identified data quality issues (missing qualifying data pre-1994, pit stop data gaps)

### 2. Exploratory Data Analysis
- **Temporal trends**: DNF rates dropped from ~50% (1950s) to ~15% (2020s)
- **Grid position impact**: Strong correlation between starting position and race outcome
- **Constructor dominance**: Top constructors consistently outperform across eras
- **Circuit variation**: Tracks exhibit distinct characteristics affecting outcomes
- **Missing data patterns**: Systematic gaps tied to regulation changes and data collection evolution

### 3. Feature Engineering
- Built **45 leakage-safe features** from pre-race information only
- Applied `shift(1)` on all rolling/expanding statistics
- Used label encoding for categorical variables with frequency-based ordering

### 4. Modeling & Evaluation
- **Temporal three-way split**: Train (pre-2018) → Validation (2018–2021) → Test (2022+)
- Hyperparameter tuning via grid search on validation set
- PR AUC selected as primary metric due to moderate class imbalance
- Ensemble methods built from top-3 individual models

### 5. Interpretability & Deployment
- SHAP analysis and permutation importance for feature explanations
- Calibration curves for probability reliability assessment
- FastAPI template for serving predictions

---

## ⚙️ Feature Engineering

All **45 features** use strictly pre-race information. Key feature categories:

### 🏁 Grid & Qualifying
| Feature | Description |
|---|---|
| `grid_position_numeric` | Numeric starting grid position |
| `grid_percentile` | Grid position as percentile of field |
| `front_row_start` | Binary: started in front row |
| `top5_start` / `top10_start` | Binary: started in top 5/10 |
| `max_position_gain` | Maximum positions gainable from grid |

### 👤 Driver Performance
| Feature | Description |
|---|---|
| `driver_prev_5_top10_rate` | Top-10 rate in last 5 races |
| `driver_prev_5_dnf_rate` | DNF rate in last 5 races |
| `driver_prev_5_avg_position` | Average finish position (last 5 races) |
| `driver_career_top10_rate` | Career-long top-10 rate |
| `driver_career_points_per_race` | Lifetime points per race |
| `driver_career_races` | Career race count |
| `driver_age` | Driver age at race time |

### 🏗️ Constructor Strength
| Feature | Description |
|---|---|
| `constructor_prev_5_top10_rate` | Constructor's top-10 rate (last 5 races) |
| `constructor_prev_5_dnf_rate` | Constructor's DNF rate (last 5 races) |
| `constructor_season_top10_rate` | Constructor's current-season top-10 rate |

### 🏟️ Circuit Characteristics
| Feature | Description |
|---|---|
| `circuit_avg_top10_rate` | Historical top-10 rate at circuit |
| `circuit_avg_dnf_rate` | Historical DNF rate at circuit |
| `circuit_position_volatility` | Position change volatility at circuit |
| `high_altitude_circuit` | Binary: high-altitude track |

### 📊 Championship & Competitive Context
| Feature | Description |
|---|---|
| `championship_points` / `championship_position` | Current standings |
| `points_gap_to_leader` | Gap to championship leader |
| `in_championship_contention` | Binary: still in title fight |
| `num_competitors` | Field size |

### 📈 Momentum & Trends
| Feature | Description |
|---|---|
| `driver_top10_streak` | Consecutive top-10 finishes |
| `driver_points_last3` | Points in last 3 races |
| `driver_position_trend` | Position improvement trend |
| `driver_avg_pit_stops` | Average pit stops per race |
| `constructor_avg_pit_time` | Constructor's average pit time |

### 🔧 Interaction & Categorical
| Feature | Description |
|---|---|
| `driver_constructor_races` | Races with current constructor |
| `driver_constructor_top10_rate` | Top-10 rate with current constructor |
| `driver_circuit_top10_rate` | Driver's top-10 rate at this circuit |
| `era_encoded` | Regulatory era indicator |
| `season_early/mid/late/end` | Season stage one-hot encoding |

---

## 📊 Models & Results

### Individual Models

Five models were trained with hyperparameter tuning on the temporal validation set:

| # | Model | Description |
|---|---|---|
| 1 | **Logistic Regression** | Linear baseline with regularization |
| 2 | **Random Forest** | Ensemble of decision trees with bagging |
| 3 | **Gradient Boosting** | Sequential boosting of weak learners |
| 4 | **MLP (Neural Network)** | Multi-layer perceptron (128→64 architecture) |
| 5 | **Baseline (Grid-only LR)** | Logistic regression using only grid position |

### Ensemble Methods

The top-3 models (Random Forest, Gradient Boosting, MLP) were combined using four ensemble strategies:

| # | Ensemble | Strategy |
|---|---|---|
| 1 | **Soft Voting** | Average predicted probabilities |
| 2 | **Hard Voting** | Majority vote on class labels |
| 3 | **Weighted Voting** | Weighted probability average |
| 4 | **Stacking** | Meta-learner on base model predictions |

### 🏆 Test Set Performance (2022+ Seasons)

| Model | ROC AUC | PR AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **Ensemble-Stacking** | **0.869** | **0.852** | **0.799** | 0.807 | 0.791 |
| Ensemble-Soft Voting | 0.869 | 0.854 | 0.798 | 0.795 | 0.800 |
| Ensemble-Weighted | 0.869 | 0.854 | 0.798 | 0.795 | 0.800 |
| Random Forest | 0.868 | 0.846 | 0.804 | 0.786 | 0.822 |
| Gradient Boosting | 0.864 | 0.852 | 0.789 | 0.809 | 0.771 |
| Logistic Regression | 0.856 | 0.848 | 0.786 | 0.792 | 0.779 |
| MLP | 0.856 | 0.835 | 0.793 | 0.777 | 0.809 |
| Baseline (Grid-only LR) | 0.790 | 0.724 | 0.757 | 0.738 | 0.778 |

> 📌 All models significantly outperform the grid-only baseline, confirming the value of the engineered feature set. Ensemble methods provide the most balanced and robust predictions.

### 📉 Visualization Outputs

The notebook generates several diagnostic plots saved in `notebook/results/final_project/`:
- `roc_curves_test.png` — ROC curves comparing all models
- `pr_curves_test.png` — Precision-Recall curves
- `confusion_matrices_test_top4.png` — Confusion matrices for top models
- `calibration_test.png` — Probability calibration curves
- `permutation_importance_top20.png` — Top-20 feature importance chart

---

## 📂 Project Structure

```
MSDS422-Final-Project-F1-Race-Prediction/
│
├── 📄 README.md
├── 📄 .gitignore
├── 📄 Assumptions_Limitations_Impact_LessonsLearned_NextSteps_CRISP-DM.pdf
│
├── 📁 data/                          # Raw F1 dataset (14 CSV tables)
│   ├── circuits.csv
│   ├── constructor_results.csv
│   ├── constructor_standings.csv
│   ├── constructors.csv
│   ├── driver_standings.csv
│   ├── drivers.csv
│   ├── lap_times.csv
│   ├── pit_stops.csv
│   ├── qualifying.csv
│   ├── races.csv
│   ├── results.csv
│   ├── seasons.csv
│   ├── sprint_results.csv
│   └── status.csv
│
├── 📁 notebook/                       # Analysis notebooks
│   ├── F1_Race_Prediction_V3.ipynb    # 🔑 Main notebook (final version)
│   ├── F1_Race_Prediction_V3.html     # Rendered HTML export
│   ├── crisp_dm.png                   # CRISP-DM diagram
│   ├── 📁 archive/                    # Previous notebook iterations
│   ├── 📁 models/                     # Serialized model artifacts
│   │   └── final_project/
│   │       └── random_forest_*/       # Saved models, encoders, feature lists
│   └── 📁 results/                    # Generated outputs
│       └── final_project/
│           ├── model_metrics_all_splits.csv
│           ├── model_and_ensemble_comparison_val_test.csv
│           ├── permutation_importance_full.csv
│           ├── shap_sample.csv / shap_values.npy
│           ├── fastapi_template.py    # Deployment template
│           └── *.png                  # Diagnostic plots
│
├── 📁 reports/                        # Written reports
│   ├── Final/                         # Final report (PDF + DOCX)
│   ├── Milestone#1/                   # Milestone 1 deliverables
│   ├── Milestone#2/                   # Milestone 2 deliverables
│   └── draft/                         # Working drafts
│
├── 📁 presentation/                   # Slide decks
│   ├── Final/                         # Final presentation (PDF + PPTX)
│   ├── Midpoint/                      # Midpoint presentation
│   └── archive/                       # Earlier versions
│
├── 📁 articles/                       # Reference papers (10 academic articles)
│
└── 📁 video/                          # Presentation recordings
    ├── F1-Sara.mp4
    ├── F1_Boqi.mp4
    └── F1_Qifan.mp4
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Clone the repository
git clone https://github.com/QifanYang17/MSDS422-Final-Project-F1-Race-Prediction.git
cd MSDS422-Final-Project-F1-Race-Prediction

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn plotly scipy shap kagglehub joblib

# Launch Jupyter
jupyter notebook notebook/F1_Race_Prediction_V3.ipynb
```

### Usage

1. **Run the notebook** — `notebook/F1_Race_Prediction_V3.ipynb` contains the complete end-to-end pipeline
2. **Data is included** — All 14 CSV files are in the `data/` directory, ready to use
3. **Results are pre-computed** — Model metrics and plots are saved in `notebook/results/final_project/`
4. **View without running** — Open `notebook/F1_Race_Prediction_V3.html` for a static rendered version

---

## 💡 Key Findings

### 1. Grid Position Is King, But Not Everything 🏁
Starting position is one of the strongest predictors, but the full feature set provides a **+7.9% ROC AUC improvement** over grid-only predictions (0.869 vs 0.790), proving that driver form, constructor strength, and context matter significantly.

### 2. Constructor Dominance Transcends Eras 🏗️
Constructor performance metrics consistently rank among the most important features across all models, reflecting F1's fundamental reality that machinery quality heavily influences outcomes.

### 3. Temporal Context Is Essential ⏳
DNF rates have dropped dramatically (from ~50% in the 1950s to ~15% in the 2020s), and scoring systems have evolved. Using temporal splits instead of random splits ensures models generalize to future unseen seasons.

### 4. Ensemble Methods Provide the Best Balance 🤖
While Random Forest achieves the highest individual recall (0.822), the Stacking Ensemble delivers the best overall balance across all metrics, combining the complementary strengths of tree-based and neural network models.

### 5. Leakage Prevention Is Non-Negotiable 🔒
Strict `shift(1)` operations on all rolling statistics and exclusive use of pre-race information ensures that model performance reflects realistic forecasting capability, not information from the future.

---

## 📚 References

The `articles/` directory contains 10 academic papers that informed this project's methodology, including research on:
- Bayesian analysis of Formula One race results
- Data-driven analysis of F1 car race outcomes
- Machine learning approaches to F1 prediction
- Statistical separation of car and driver effects in racing

**Data Source:**
> Vopani. *Formula 1 World Championship (1950–2024).* Kaggle, 2025. [Link](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)

---

## 👥 Team

| Name | Role |
|---|---|
| **Sara Alsiyat** | Team Member |
| **Boqi Niu** | Team Member |
| **Qifan Yang** | Team Member |

**Course:** MSDS 422 – Practical Machine Learning
**Institution:** Northwestern University

---

<p align="center">
  <em>Built with ❤️ and data from 74 years of Formula One racing</em>
</p>
