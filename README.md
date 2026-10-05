# 📊 Customer Churn Prediction & Analytics

> **"The project analyzes historical telecom customer data to identify factors associated with churn, predicts customers at risk of leaving using machine learning, and provides business insights and retention recommendations through an interactive dashboard."**

---

## 📌 Table of Contents

1. [What is This Project?](#-what-is-this-project)
2. [Why is This Project Useful?](#-why-is-this-project-useful)
3. [Dataset Description](#-dataset-description)
4. [Project Pipeline (How It Works)](#-project-pipeline-how-it-works)
5. [Technologies Used](#️-technologies-used)
6. [Project Structure](#-project-structure)
7. [Step-by-Step: How to Run the Project](#-step-by-step-how-to-run-the-project)
8. [Step-by-Step: What Each Module Does](#-step-by-step-what-each-module-does)
9. [ML Model Results](#-ml-model-results)
10. [Risk Analysis](#️-risk-analysis)
11. [Key Findings & Insights](#-key-findings--insights)
12. [Business Recommendations](#-business-recommendations)
13. [Dashboard](#-dashboard)
14. [Sample Output Screenshots](#-sample-output-screenshots)
15. [Future Improvements](#-future-improvements)

---

## 🎯 What is This Project?

A telecom company (like Jio, Airtel, or Vodafone) has thousands of customers. Every month, some customers **stop using their services** — this is called **Customer Churn**.

This project uses the company's **past customer data** to:

1. **Find patterns** → Why are customers leaving?
2. **Predict** → Which customers are likely to leave next?
3. **Recommend** → What can the company do to retain them?

### Simple Example

```
Customer A
├── Tenure: 2 months
├── Contract: Month-to-month
├── Monthly Bill: ₹1,500
├── No Tech Support
└── Churn Probability: 85% → 🔴 HIGH RISK

Action: The company can contact this customer before they leave,
        offer a better plan, or provide free tech support.
```

---

## 💡 Why is This Project Useful?

This is a **Business Analytics** project. It's not just about prediction — it answers real business questions:

| Business Question | How This Project Answers It |
|---|---|
| Why are customers leaving? | Churn Pattern Analysis (Step 3) |
| Which customers are most at risk? | Risk Analysis (Step 6) |
| What factors are associated with churn? | EDA + Feature Importance (Steps 2, 5) |
| Where should the company focus retention efforts? | Business Recommendations (Step 7) |
| How can we visualize all of this? | Interactive Dashboard (Step 8) |

**Real-world use:**

```
Past customer data → Identify churn patterns → Predict risky customers → Take retention action → Reduce customer loss
```

---

## 📋 Dataset Description

**File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`  
**Source:** IBM Sample Telco Customer Churn Dataset  
**Rows:** 7,044 customers  
**Columns:** 21 features  

| Column | Type | Description |
|--------|------|-------------|
| `customerID` | String | Unique customer identifier |
| `gender` | Categorical | Male / Female |
| `SeniorCitizen` | Binary | 0 = No, 1 = Yes |
| `Partner` | Categorical | Has a partner? Yes / No |
| `Dependents` | Categorical | Has dependents? Yes / No |
| `tenure` | Numeric | Number of months with the company (0–72) |
| `PhoneService` | Categorical | Has phone service? Yes / No |
| `MultipleLines` | Categorical | Yes / No / No phone service |
| `InternetService` | Categorical | DSL / Fiber optic / No |
| `OnlineSecurity` | Categorical | Yes / No / No internet service |
| `OnlineBackup` | Categorical | Yes / No / No internet service |
| `DeviceProtection` | Categorical | Yes / No / No internet service |
| `TechSupport` | Categorical | Yes / No / No internet service |
| `StreamingTV` | Categorical | Yes / No / No internet service |
| `StreamingMovies` | Categorical | Yes / No / No internet service |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `PaperlessBilling` | Categorical | Yes / No |
| `PaymentMethod` | Categorical | Electronic check / Mailed check / Bank transfer / Credit card |
| `MonthlyCharges` | Numeric | Monthly bill amount |
| `TotalCharges` | Numeric | Total amount charged to the customer |
| **`Churn`** | **Target** | **Yes = Left the company / No = Still a customer** |

---

## 🔄 Project Pipeline (How It Works)

```
Telco Customer Data (7,044 customers, 21 features)
        │
        ▼
┌─────────────────────────────────────────────────┐
│  STEP 1: Data Cleaning                          │
│  Handle missing values, fix data types,         │
│  remove customerID, convert categories          │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│  STEP 2: Business Analysis (EDA)                │
│  11 charts: churn by contract, tenure,          │
│  charges, payment method, services, etc.        │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│  STEP 3: Churn Pattern Analysis                 │
│  Identify top churn drivers, analyze            │
│  service combinations, rank risk factors        │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│  STEP 4: Customer Segmentation                  │
│  Group by tenure & charges, profile each        │
│  segment, cross-analysis heatmap                │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│  STEP 5: ML Churn Prediction                    │
│  Train 4 models (LR, DT, RF, XGBoost),         │
│  evaluate, select best, save model              │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│  STEP 6: Risk Analysis                          │
│  Score every customer with churn probability,   │
│  categorize into Low / Medium / High Risk       │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│  STEP 7: Business Recommendations               │
│  6 data-driven retention strategies             │
│  based on analysis findings                     │
└────────────────────┬────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────┐
│  STEP 8: Interactive Dashboard                  │
│  HTML dashboard with KPIs, charts, risk table,  │
│  and recommendations — light pastel theme       │
└─────────────────────────────────────────────────┘
```

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.14+ | Core programming language |
| **Pandas** | 2.0+ | Data loading, manipulation, and analysis |
| **NumPy** | 1.24+ | Numerical computations |
| **Matplotlib** | 3.7+ | Chart creation and visualization |
| **Seaborn** | 0.13+ | Statistical data visualization |
| **Scikit-learn** | 1.3+ | ML models, preprocessing, evaluation |
| **XGBoost** | 2.0+ | Gradient boosting classifier |
| **Joblib** | 1.3+ | Model saving/loading |
| **HTML/CSS/JS** | — | Interactive dashboard |

---

## 📁 Project Structure

```
Customer Churn Prediction/
│
├── main.py                              # 🚀 Run this to execute the full pipeline
├── requirements.txt                     # Python dependencies
├── README.md                            # This documentation file
├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Original dataset (input)
│
├── src/                                 # Source code (7 modules)
│   ├── __init__.py                      # Python package marker
│   ├── data_cleaning.py                 # Step 1: Cleans raw data
│   ├── eda_analysis.py                  # Step 2: Creates 11 EDA charts
│   ├── churn_patterns.py               # Step 3: Identifies churn drivers
│   ├── customer_segmentation.py        # Step 4: Segments customers
│   ├── ml_prediction.py                # Step 5: Trains 4 ML models
│   ├── risk_analysis.py                # Step 6: Scores customer risk
│   └── recommendations.py             # Step 7: Generates recommendations
│
├── data/                                # Generated data files
│   ├── telco_churn_cleaned.csv          # Cleaned dataset (after Step 1)
│   └── customer_risk_scores.csv         # All customers with risk scores (after Step 6)
│
├── models/                              # Saved ML models
│   └── churn_model.pkl                  # Best trained model + scaler + features
│
├── reports/                             # Analysis outputs
│   ├── recommendations.txt             # Business recommendations text file
│   └── figures/                         # All 25 generated charts (PNG)
│       ├── churn_distribution.png
│       ├── churn_by_contract.png
│       ├── churn_by_tenure.png
│       ├── churn_by_monthly_charges.png
│       ├── churn_by_payment_method.png
│       ├── churn_by_internet_service.png
│       ├── churn_by_tech_support.png
│       ├── churn_by_gender.png
│       ├── churn_by_senior_citizen.png
│       ├── churn_by_dependents_partner.png
│       ├── correlation_heatmap.png
│       ├── top_churn_factors.png
│       ├── service_combination_churn.png
│       ├── segment_tenure_churn.png
│       ├── segment_charges_churn.png
│       ├── segment_cross_analysis.png
│       ├── segment_profiles.png
│       ├── model_comparison.png
│       ├── confusion_matrix.png
│       ├── roc_curves.png
│       ├── feature_importance.png
│       ├── risk_distribution.png
│       ├── risk_by_tenure.png
│       ├── risk_by_charges.png
│       └── risk_probability_histogram.png
│
└── dashboard/                           # Interactive dashboard
    └── churn_dashboard.html             # Open in any browser
```

---

## 🚀 Step-by-Step: How to Run the Project

### Prerequisites

Make sure you have **Python 3.8+** installed on your system.

```bash
python --version
```

### Step 1: Open the Project Folder

Open the `Customer Churn Prediction` folder in VS Code. Select **Terminal → New Terminal**. VS Code opens the terminal in the project folder automatically, so you do not need to run a `cd` command.

### Step 2: Install Dependencies

In the terminal, run:

```powershell
python -m pip install -r requirements.txt
```

This installs: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `joblib`

### Step 3: Make Sure the Dataset is Present

Verify that `WA_Fn-UseC_-Telco-Customer-Churn.csv` is in the project root folder.

### Step 4: Run the Full Pipeline

```bash
python main.py
```

This single command runs **all 8 steps** automatically:

```
Step 1 → Data Cleaning          (~1s)
Step 2 → EDA Charts             (~20s)
Step 3 → Churn Patterns         (~1s)
Step 4 → Customer Segmentation  (~1s)
Step 5 → ML Training            (~5s)
Step 6 → Risk Scoring           (~2s)
Step 7 → Recommendations        (~0.1s)
Step 8 → Dashboard Generation   (~0.5s)
─────────────────────────────────────
Total                            ~30s
```

### Step 5: View the Dashboard

Open the generated dashboard in your browser:

```powershell
Start-Process .\dashboard\churn_dashboard.html
```

You can also open `dashboard/churn_dashboard.html` from VS Code's Explorer.

### Optional: Run Individual Steps

You can also run each module individually:

```bash
python -m src.data_cleaning
python -m src.eda_analysis
python -m src.churn_patterns
python -m src.customer_segmentation
python -m src.ml_prediction
python -m src.risk_analysis
python -m src.recommendations
```

### Step 8 (Optional): Use in Power BI

Import `data/customer_risk_scores.csv` into Power BI Desktop to create your own interactive visuals.

---

## 📝 Step-by-Step: What Each Module Does

### Step 1: Data Cleaning (`src/data_cleaning.py`)

**Purpose:** Prepare raw data for analysis.

| Task | Details |
|------|---------|
| Load dataset | Reads the original CSV (7,044 rows, 21 columns) |
| Fix `TotalCharges` | Converts from string to float; replaces blank values with 0 |
| Drop `customerID` | Removed since it's not useful for analysis (saved separately for later) |
| Convert `SeniorCitizen` | Changed from 0/1 to "No"/"Yes" for readability |
| Check duplicates | Scans and removes any duplicate rows |
| Save cleaned data | Outputs to `data/telco_churn_cleaned.csv` |

**Input:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`  
**Output:** `data/telco_churn_cleaned.csv`

---

### Step 2: Exploratory Data Analysis (`src/eda_analysis.py`)

**Purpose:** Visualize data to understand churn patterns. Answers key business questions.

| Chart | Business Question Answered |
|-------|---------------------------|
| `churn_distribution.png` | What % of customers churn? → **26.4%** |
| `churn_by_contract.png` | Which contract type has more churn? → **Month-to-month** |
| `churn_by_tenure.png` | Does tenure affect churn? → **Yes, shorter tenure = higher churn** |
| `churn_by_monthly_charges.png` | Do higher charges lead to more churn? → **Yes** |
| `churn_by_payment_method.png` | Which payment method has higher churn? → **Electronic check** |
| `churn_by_tech_support.png` | Does tech support reduce churn? → **Yes, significantly** |
| `churn_by_internet_service.png` | Which internet service has more churn? → **Fiber optic** |
| `churn_by_gender.png` | Does gender affect churn? → **Minimal difference** |
| `churn_by_senior_citizen.png` | Do senior citizens churn more? → **Yes** |
| `churn_by_dependents_partner.png` | Do partner/dependents affect churn? → **Yes, singles churn more** |
| `correlation_heatmap.png` | How are numeric features correlated? |

**Input:** `data/telco_churn_cleaned.csv`  
**Output:** 11 chart PNGs in `reports/figures/`

---

### Step 3: Churn Pattern Analysis (`src/churn_patterns.py`)

**Purpose:** Dig deeper — identify the **top churn drivers** and dangerous service combinations.

**What it does:**
- Computes churn rate for **every value of every categorical feature**
- Ranks them from highest to lowest churn rate
- Analyzes dangerous **service combinations**:
  - Fiber optic + No OnlineSecurity + No TechSupport → **54.9% churn**
  - Month-to-month + Electronic check → **53.6% churn**
- Prints **5 key findings** with data-backed evidence

**Input:** `data/telco_churn_cleaned.csv`  
**Output:** `top_churn_factors.png`, `service_combination_churn.png`

---

### Step 4: Customer Segmentation (`src/customer_segmentation.py`)

**Purpose:** Group customers into meaningful segments and profile each group.

**Tenure Segments:**

| Segment | Tenure | Count | Churn Rate |
|---------|--------|-------|------------|
| New | 0–12 months | 2,164 | **47.4%** |
| Medium | 13–24 months | 1,024 | 28.7% |
| Established | 25–48 months | 1,594 | 20.4% |
| Loyal | 49–72 months | 2,239 | **9.5%** |

**Charge Segments:**

| Segment | Charges | Count | Churn Rate |
|---------|---------|-------|------------|
| Low | Lowest 25% | 1,761 | 11.1% |
| Medium | 25–50% | 1,754 | 24.5% |
| High | 50–75% | 1,756 | **37.4%** |
| Premium | Top 25% | 1,750 | 32.9% |

**Cross-Analysis:** Heatmap of Tenure Group × Contract Type → Churn Rate

**Input:** `data/telco_churn_cleaned.csv`  
**Output:** `segment_tenure_churn.png`, `segment_charges_churn.png`, `segment_cross_analysis.png`, `segment_profiles.png`

---

### Step 5: ML Churn Prediction (`src/ml_prediction.py`)

**Purpose:** Train machine learning models to predict churn.

**Process:**
1. Convert target (`Churn`) to binary (Yes=1, No=0)
2. One-hot encode all categorical features using `pd.get_dummies()`
3. Scale numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) with StandardScaler
4. Split data: 80% training / 20% testing (stratified)
5. Train 4 models
6. Evaluate and compare all models
7. Select the best model and save it

**Models Trained:**

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | **80.2%** | **66.0%** | **52.2%** | **58.3%** | **0.840** ✅ |
| XGBoost | 79.5% | 64.5% | 50.3% | 56.5% | 0.835 |
| Decision Tree | 78.7% | 65.2% | 42.2% | 51.2% | 0.821 |
| Random Forest | 78.6% | 63.4% | 45.2% | 52.8% | 0.819 |

**Best Model:** Logistic Regression (highest AUC-ROC = 0.840)

**Top 5 Most Important Features:**
1. `tenure` — how long the customer has been with the company
2. `Contract_Two year` — two-year contract (reduces churn)
3. `InternetService_Fiber optic` — fiber optic users churn more
4. `TotalCharges` — lifetime spending amount
5. `Contract_One year` — one-year contract (reduces churn)

**Input:** `data/telco_churn_cleaned.csv`  
**Output:** `models/churn_model.pkl`, `model_comparison.png`, `confusion_matrix.png`, `roc_curves.png`, `feature_importance.png`

---

### Step 6: Risk Analysis (`src/risk_analysis.py`)

**Purpose:** Use the best model to score **every customer** with a churn probability and categorize their risk.

**Risk Categories:**

| Churn Probability | Risk Level | Color |
|-------------------|------------|-------|
| 0–30% | 🟢 Low Risk | Green |
| 30–60% | 🟡 Medium Risk | Yellow |
| 60–100% | 🔴 High Risk | Red |

**Risk Distribution (Results):**

| Risk Level | Count | Percentage |
|-----------|-------|------------|
| 🟢 Low Risk | 4,366 | 62.2% |
| 🟡 Medium Risk | 1,637 | 23.3% |
| 🔴 High Risk | 1,018 | **14.5%** |

**High-Risk Customer Profile:**
- Average tenure: **8.8 months** (very new customers)
- Most common contract: **Month-to-month**
- Most common internet: **Fiber optic**
- Average monthly charges: **$82.09**

**Input:** `data/telco_churn_cleaned.csv`, `models/churn_model.pkl`  
**Output:** `data/customer_risk_scores.csv`, `risk_distribution.png`, `risk_by_tenure.png`, `risk_by_charges.png`, `risk_probability_histogram.png`

---

### Step 7: Business Recommendations (`src/recommendations.py`)

**Purpose:** Generate **data-driven** retention recommendations (not random suggestions — each one is backed by the analysis).

See [Business Recommendations](#-business-recommendations) section below for the full list.

**Input:** `data/customer_risk_scores.csv`  
**Output:** `reports/recommendations.txt`

---

### Step 8: Dashboard Generation (inside `main.py`)

**Purpose:** Create a single-file interactive HTML dashboard.

**Features:**
- 5 KPI cards (Total Customers, Churned, Churn Rate, Avg Charges, High-Risk Count)
- Tabbed navigation (Overview | EDA | ML Models | Risk | Recommendations)
- All 25 charts embedded as base64 images
- Risk summary with colored badges
- Top 10 high-risk customers table
- Light pastel color theme

**Input:** All generated charts + data files  
**Output:** `dashboard/churn_dashboard.html`

---

## 🤖 ML Model Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Selected? |
|-------|----------|-----------|--------|----------|---------|-----------|
| **Logistic Regression** | **80.2%** | **66.0%** | **52.2%** | **58.3%** | **0.840** | ✅ Best |
| XGBoost | 79.5% | 64.5% | 50.3% | 56.5% | 0.835 | |
| Decision Tree | 78.7% | 65.2% | 42.2% | 51.2% | 0.821 | |
| Random Forest | 78.6% | 63.4% | 45.2% | 52.8% | 0.819 | |

### What These Metrics Mean

| Metric | Meaning | Our Best Score |
|--------|---------|----------------|
| **Accuracy** | % of correct predictions overall | 80.2% |
| **Precision** | Of those predicted as churners, how many actually churned? | 66.0% |
| **Recall** | Of all actual churners, how many did we catch? | 52.2% |
| **F1-Score** | Balance between Precision and Recall | 58.3% |
| **AUC-ROC** | Model's ability to distinguish churners from non-churners (0.5 = random, 1.0 = perfect) | **0.840** |

### Why Logistic Regression Won

Logistic Regression achieved the **highest AUC-ROC (0.840)**, meaning it best separates churners from non-churners across all probability thresholds. It also has the highest recall, meaning it catches more actual churners.

---

## ⚠️ Risk Analysis

### Risk Categories

Every customer is scored with a **churn probability** (0% to 100%) and placed into a risk category:

```
Customer A → 87% churn probability → 🔴 HIGH RISK
Customer B → 31% churn probability → 🟡 MEDIUM RISK
Customer C →  8% churn probability → 🟢 LOW RISK
```

### Risk Distribution

| Risk Level | Count | Percentage | Action |
|-----------|-------|------------|--------|
| 🟢 Low Risk (0–30%) | 4,366 | 62.2% | Monitor |
| 🟡 Medium Risk (30–60%) | 1,637 | 23.3% | Proactive outreach |
| 🔴 High Risk (60–100%) | **1,018** | **14.5%** | **Immediate intervention** |

### Typical High-Risk Customer

- Joined **less than 9 months ago**
- On a **month-to-month contract**
- Uses **Fiber optic** internet
- Pays about **$82/month**
- **No Tech Support** or **Online Security**
- Pays via **Electronic check**

---

## 🔍 Key Findings & Insights

### Top 5 Churn Drivers

| Rank | Factor | Churn Rate | Business Insight |
|------|--------|------------|------------------|
| 1 | Fiber optic + No Security/Tech Support | **54.9%** | Fiber customers without add-on services are extremely vulnerable |
| 2 | Month-to-month + Electronic check | **53.6%** | This combination is the highest-risk profile |
| 3 | Electronic check payment | **45.1%** | Manual payment method correlates with higher churn |
| 4 | Month-to-month contract | **42.6%** | No commitment = easy to leave |
| 5 | No Tech Support | **41.5%** | Lack of support drives dissatisfaction |

### Factors That Reduce Churn

| Factor | Churn Rate | Insight |
|--------|------------|---------|
| Two-year contract | **2.6%** | Long commitment = very low churn |
| One-year contract | **11.3%** | Still much lower than month-to-month |
| Loyal customers (49-72 months) | **9.5%** | Once past 4 years, customers stay |
| Low monthly charges | **11.1%** | Affordable plans retain customers |

### Key Observations

1. **Tenure is the #1 predictor** — the longer a customer stays, the less likely they are to churn
2. **Contract type matters hugely** — month-to-month customers churn at 42.6% vs 2.6% for two-year contracts
3. **Gender has minimal impact** — male (26.1%) vs female (26.8%), almost identical
4. **Service bundles protect** — customers with Tech Support, Online Security, and Online Backup churn less
5. **Price sensitivity exists** — churned customers pay $74.60 avg vs $61.34 for retained customers

---

## 💡 Business Recommendations

All recommendations are **data-driven** — each one is backed by findings from the analysis:

### Recommendation #1: Contract Migration Program
- **Finding:** Month-to-month contracts have the highest churn rate at **42.6%** (3,853 customers)
- **Action:** Offer incentives (first month free, discounts) to switch to 1-year or 2-year contracts
- **Expected Benefit:** Reduce churn by increasing customer commitment

### Recommendation #2: New Customer Onboarding Program
- **Finding:** New customers (0-1 year) have the highest churn rate at **47.4%** (2,164 customers)
- **Action:** Implement targeted onboarding programs with early touchpoints and support
- **Expected Benefit:** Improve early retention and user experience

### Recommendation #3: Auto-Pay Migration
- **Finding:** Electronic check users have the highest churn rate at **45.1%** (2,359 customers)
- **Action:** Encourage migration to automatic payments (credit card/bank transfer) via small discounts
- **Expected Benefit:** Reduce payment friction and involuntary churn

### Recommendation #4: Tech Support Bundle
- **Finding:** Customers without Tech Support churn at **41.5%** (3,465 customers)
- **Action:** Bundle Tech Support with core internet services or offer a free trial
- **Expected Benefit:** Increase product stickiness and customer satisfaction

### Recommendation #5: Fiber Optic Service Improvement
- **Finding:** Fiber optic customers have the highest churn rate at **41.8%** (3,090 customers)
- **Action:** Investigate Fiber Optic service quality, reliability, and pricing
- **Expected Benefit:** Address service issues and retain high-value customers

### Recommendation #6: Pricing Review
- **Finding:** Churned customers pay an average of **$74.60** vs **$61.34** for retained customers
- **Action:** Conduct a pricing review and offer flexible, personalized plans
- **Expected Benefit:** Prevent cost-driven churn

---

## 📊 Dashboard

The interactive dashboard is a **standalone HTML file** — no server needed. Just open it in any browser.

### Dashboard Features

| Feature | Description |
|---------|-------------|
| **KPI Cards** | Total Customers, Churned, Churn Rate, Avg Monthly Charges, High-Risk Count |
| **Tab: Overview** | KPIs + Risk Summary badges + Top 10 High-Risk Customers table |
| **Tab: EDA Analysis** | All exploratory charts (churn by contract, tenure, charges, etc.) |
| **Tab: ML Models** | Model comparison, confusion matrices, ROC curves, feature importance |
| **Tab: Risk Analysis** | Risk distribution, risk by tenure, risk by charges, probability histogram |
| **Tab: Recommendations** | All 6 data-driven business recommendations |

### How to Open

```bash
# Windows
start dashboard/churn_dashboard.html

# Mac
open dashboard/churn_dashboard.html

# Linux
xdg-open dashboard/churn_dashboard.html
```

### For Power BI

Import `data/customer_risk_scores.csv` into Power BI Desktop to create your own custom interactive visuals with filters and slicers.

---

## 📸 Sample Output Screenshots

### Charts Generated (25 total)

| Category | Charts |
|----------|--------|
| **EDA** (11) | Churn distribution, contract, tenure, charges, payment method, tech support, internet service, gender, senior citizen, partner/dependents, correlation heatmap |
| **Patterns** (2) | Top churn factors, service combination churn |
| **Segmentation** (4) | Tenure churn, charges churn, cross-analysis heatmap, segment profiles |
| **ML Models** (4) | Model comparison, confusion matrices, ROC curves, feature importance |
| **Risk** (4) | Risk distribution, risk by tenure, risk by charges, probability histogram |

All charts use a **light pastel color palette** for a clean, professional look.

---

## 🔮 Future Improvements

1. **Deep Learning Models** — Try neural networks for potentially better prediction
2. **SMOTE Oversampling** — Handle class imbalance (26% churn vs 74% no-churn)
3. **Feature Engineering** — Create new features like charges-per-month-of-tenure ratio
4. **Real-time Scoring** — Build an API that scores new customers as they sign up
5. **A/B Testing Framework** — Test which retention strategies work best
6. **Cost-Benefit Analysis** — Calculate ROI of retention campaigns
7. **Customer Lifetime Value (CLV)** — Prioritize retention for high-value customers

---

## 👨‍💻 Author

Customer Churn Prediction — Business Analytics Project

---

*Built with Python, Scikit-learn, XGBoost, and lots of data-driven insights.* 📊
