# 📊 Exploratory Data Analysis — Student Performance in Portugal

**Aix-Marseille School of Economics | MAG1 — Semestre 2 | Programmation pour la Data**
**Academic Year 2025-2026**

---

## Project Overview

This project conducts a full **Exploratory Data Analysis** (EDA) on the *Student Performance Dataset* (Cortez & Silva, 2008 — UCI Machine Learning Repository). It examines the socio-demographic, behavioural, and family factors that influence the final grades of students in **Mathematics** and **Portuguese** across two secondary schools in Portugal.

Two separate **OLS regression models** are estimated — one per subject — to quantify the marginal effect of each predictor on the final grade G3, while deliberately excluding G1 and G2 (endogenous) to focus on actionable, exogenous determinants.

---

## Project Structure

```
├── EDA_Project.ipynb   # Main notebook (8 sections, fully documented)
├── Dashboard_app.py          # Interactive Dash dashboard
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Dataset

| File | Subject | N students | N variables |
|------|---------|-----------|-------------|
| `student_mat.csv` | Mathematics | 395 | 33 |
| `student_por.csv` | Portuguese  | 649 | 33 |

**Source:** [UCI Machine Learning Repository — Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
**Reference:** P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance*. 2008.

**Target variable:** `G3` — Final grade (0–20)
**Key predictors:** failures, studytime, schoolsup, sex, higher, goout, romantic, health, school, Medu, Fedu, absences

> G1 and G2 are excluded from regression models (r > 0.90 with G3 — endogenous by construction).

---

##  Notebook Structure

| Section | Content |
|---------|---------|
| **1. Introduction** | Context, objectives, research question, dataset description |
| **2. Libraries & Loading** | Imports, data loading, merged dataset |
| **3. Data Understanding** | Variable classification, descriptive statistics, G3 distribution |
| **4. Data Preparation** | Missing values, duplicates, outlier detection, feature engineering |
| **5. Bivariate Analysis** | Mann-Whitney U, Chi², Cramér's V, effect size synthesis |
| **6. Correlation Analysis** | Pearson matrix, multicollinearity identification |
| **7. OLS Regression** | Dual OLS (MAT vs POR), VIF |
| **8. Conclusion** | Key findings, recommendations, limitations, perspectives |

---

## ⚙️ Installation


### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the notebook

```bash
jupyter notebook EDA_Project.ipynb
```

### 3. Run the interactive dashboard

```bash
python SK_app.py
```

Then open your browser at: **http://127.0.0.1:8051**

---

### OLS Regression — Mathematics (n=395)

| Predictor | Coefficient | p-value |
|-----------|------------|---------|
| `failures` | −1.72 | < 0.001 *** |
| `goout` | −0.59 | < 0.01 ** |
| `sex_M` | +1.26 | < 0.05 * |
| `schoolsup_yes` | −1.35 | < 0.05 * |
| `romantic_yes` | −1.09 | < 0.05 * |

**R² = 0.276 | Adj. R² = 0.196 | R² CV = 0.026**

### OLS Regression — Portuguese (n=649)

| Predictor | Coefficient | p-value |
|-----------|------------|---------|
| `higher_yes` | +1.73 | < 0.001 *** |
| `failures` | −1.41 | < 0.001 *** |
| `schoolsup_yes` | −1.31 | < 0.001 *** |
| `school_MS` | −1.20 | < 0.001 *** |
| `studytime` | +0.41 | < 0.01 ** |
| `sex_M` | −0.63 | < 0.05 * |
| `health` | −0.19 | < 0.05 * |

**R² = 0.360 | Adj. R² = 0.319 | R² CV = 0.246**

### Main Takeaways

- **`failures`** is the only significant predictor in **both subjects** — each additional past failure costs 1.4 to 1.7 grade points.
- The Portuguese model is more stable (adj. R²=0.319 vs 0.196), suggesting socio-demographic factors explain language performance better than mathematical performance.
- **Gender gap**: `sex_M` is positive in Mathematics (+1.26) but negative in Portuguese (−0.63), consistent with the educational literature.
- **`higher_yes`** (aspiration to higher education) is the strongest positive driver in Portuguese (β=+1.73***).

---

## 📊 Dashboard

The `SK_app.py` file launches an interactive **Dash** application that allows users to:
- Explore the distribution of any variable
- Visualise relationships between predictors and G3 / Pass/Fail
- Filter by school, subject, sex, and other categorical variables
- Export filtered data to Excel

```bash
python SK_app.py   # → http://127.0.0.1:8051
```

---

##  Known Limitations

- Data from **two schools only** (2006–2008, Portugal) — limited external validity
- **G3 = 0** may reflect exam absence or dropout, not actual zero performance
- OLS assumes linearity — non-linear effects are not captured
- **Heteroskedasticity** detected in the Portuguese model → corrected via HC3 robust standard errors
- Cross-sectional design — no causal inference possible

---

## Reference

> Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance*. In A. Brito and J. Teixeira (Eds.), Proceedings of 5th Annual Future Business Technology Conference (FUBUTEC 2008), Porto, Portugal. EUROSIS.


