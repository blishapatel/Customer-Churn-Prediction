"""
ChurnIQ — Customer Churn Analytics & Retention Decision Support System
Flask Web Application
"""

import os
import sys
import io
import base64
import pandas as pd
import numpy as np
from flask import (
    Flask, render_template, request, jsonify, send_from_directory, url_for
)

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))
from src.prediction_engine import ChurnPredictor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

os.makedirs('uploads', exist_ok=True)

# ── Load predictor & data at startup ─────────────────────────
predictor = None
risk_df = None
cleaned_df = None


def get_predictor():
    global predictor
    if predictor is None:
        predictor = ChurnPredictor('models/churn_model.pkl')
    return predictor


def get_risk_data():
    global risk_df
    if risk_df is None:
        path = 'data/customer_risk_scores.csv'
        if os.path.exists(path):
            risk_df = pd.read_csv(path)
    return risk_df


def get_cleaned_data():
    global cleaned_df
    if cleaned_df is None:
        path = 'data/telco_churn_cleaned.csv'
        if os.path.exists(path):
            cleaned_df = pd.read_csv(path)
    return cleaned_df


# ── Chart serving ─────────────────────────────────────────────
@app.route('/chart/<filename>')
def chart(filename):
    return send_from_directory('reports/figures', filename)


# ── PAGE ROUTES ───────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('home.html', active_page='home')


@app.route('/predict')
def predict():
    return render_template('predict.html', active_page='predict')


@app.route('/bulk')
def bulk():
    return render_template('bulk.html', active_page='bulk')


@app.route('/analytics')
def analytics():
    df = get_cleaned_data()
    risk = get_risk_data()

    stats = {}
    if df is not None:
        churn_col = df['Churn']
        if churn_col.dtype == 'object':
            churned = (churn_col == 'Yes').sum()
        else:
            churned = int(churn_col.sum())

        total = len(df)
        stats = {
            'total_customers': f'{total:,}',
            'churned': f'{churned:,}',
            'churn_rate': round(churned / total * 100, 1),
            'avg_monthly': round(df['MonthlyCharges'].mean(), 2),
            'avg_total': round(df['TotalCharges'].mean(), 2),
            'revenue_at_risk': f"{churned * df['MonthlyCharges'].mean() * 12:,.0f}",
        }

    charts = [
        {'title': 'Churn Distribution',
         'filename': 'churn_distribution.png',
         'insight': '26.4% of customers have churned — about 1 in 4 customers leaves the company.'},
        {'title': 'Churn by Contract Type',
         'filename': 'churn_by_contract.png',
         'insight': 'Month-to-month contracts have dramatically higher churn (42.6%) vs Two-year (2.6%).'},
        {'title': 'Churn by Tenure',
         'filename': 'churn_by_tenure.png',
         'insight': 'New customers (0-12 months) churn the most. Retention improves significantly after the first year.'},
        {'title': 'Churn by Monthly Charges',
         'filename': 'churn_by_monthly_charges.png',
         'insight': 'Customers with higher monthly charges churn more — churned customers pay $74.60 avg vs $61.34 retained.'},
        {'title': 'Churn by Payment Method',
         'filename': 'churn_by_payment_method.png',
         'insight': 'Electronic check users churn at 45.1% — much higher than auto-pay methods (~15-17%).'},
        {'title': 'Churn by Internet Service',
         'filename': 'churn_by_internet_service.png',
         'insight': 'Fiber optic customers churn at 41.8%, significantly more than DSL (19.0%).'},
        {'title': 'Churn by Tech Support',
         'filename': 'churn_by_tech_support.png',
         'insight': 'Customers without Tech Support churn at 41.5% vs 15.2% with support.'},
        {'title': 'Churn by Gender',
         'filename': 'churn_by_gender.png',
         'insight': 'Gender has minimal impact on churn — Male (26.1%) vs Female (26.8%).'},
        {'title': 'Churn by Senior Citizen',
         'filename': 'churn_by_senior_citizen.png',
         'insight': 'Senior citizens churn at 41.6% compared to 23.6% for non-seniors.'},
        {'title': 'Churn by Partner & Dependents',
         'filename': 'churn_by_dependents_partner.png',
         'insight': 'Customers without partners or dependents show higher churn rates.'},
        {'title': 'Correlation Heatmap',
         'filename': 'correlation_heatmap.png',
         'insight': 'Tenure and TotalCharges are strongly correlated. MonthlyCharges has a positive correlation with churn.'},
        {'title': 'Top Churn Factors',
         'filename': 'top_churn_factors.png',
         'insight': 'Electronic check payment, month-to-month contract, and fiber optic service are the top churn drivers.'},
        {'title': 'Service Combination Impact',
         'filename': 'service_combination_churn.png',
         'insight': 'Fiber optic without security/tech support has 54.9% churn — the highest risk combination.'},
        {'title': 'Churn by Tenure Segment',
         'filename': 'segment_tenure_churn.png',
         'insight': 'New customers (0-12 months) churn at 47.4% — nearly 5x the rate of loyal customers (9.5%).'},
        {'title': 'Churn by Charge Segment',
         'filename': 'segment_charges_churn.png',
         'insight': 'High charge customers (top 50%) show significantly elevated churn rates.'},
        {'title': 'Tenure × Contract Heatmap',
         'filename': 'segment_cross_analysis.png',
         'insight': 'New customers on month-to-month contracts have the highest churn rate at 51.3%.'},
    ]

    # Filter out charts that don't have generated image files
    charts = [c for c in charts if os.path.exists(f'reports/figures/{c["filename"]}')]

    return render_template('analytics.html', active_page='analytics',
                           stats=stats, charts=charts)


@app.route('/risk')
def risk():
    df = get_risk_data()

    risk_stats = {'high_count': 0, 'medium_count': 0, 'low_count': 0,
                  'high_pct': 0, 'medium_pct': 0, 'low_pct': 0}
    high_risk_customers = []
    profile = {
        'avg_tenure': 'N/A', 'common_contract': 'N/A',
        'common_internet': 'N/A', 'avg_charges': 'N/A'
    }

    if df is not None and 'Risk_Category' in df.columns:
        total = len(df)
        for cat, key in [('High Risk', 'high'), ('Medium Risk', 'medium'), ('Low Risk', 'low')]:
            count = len(df[df['Risk_Category'] == cat])
            risk_stats[f'{key}_count'] = count
            risk_stats[f'{key}_pct'] = round(count / total * 100, 1) if total else 0

        # High-risk profile
        hr = df[df['Risk_Category'] == 'High Risk']
        if len(hr) > 0:
            profile['avg_tenure'] = f"{hr['tenure'].mean():.1f} months"
            if 'Contract' in hr.columns:
                profile['common_contract'] = hr['Contract'].mode().iloc[0] if len(hr['Contract'].mode()) > 0 else 'N/A'
            if 'InternetService' in hr.columns:
                profile['common_internet'] = hr['InternetService'].mode().iloc[0] if len(hr['InternetService'].mode()) > 0 else 'N/A'
            profile['avg_charges'] = f"${hr['MonthlyCharges'].mean():.2f}"

        # Table data — top 50 high-risk
        display_cols = ['customerID', 'tenure', 'Contract', 'MonthlyCharges',
                        'InternetService', 'TechSupport', 'Churn_Probability', 'Risk_Category']
        avail = [c for c in display_cols if c in df.columns]
        top = df.nlargest(50, 'Churn_Probability')[avail]
        high_risk_customers = top.to_dict('records')

    risk_charts = [
        {'title': 'Risk Category Distribution', 'filename': 'risk_distribution.png'},
        {'title': 'Tenure by Risk Category', 'filename': 'risk_by_tenure.png'},
        {'title': 'Monthly Charges by Risk Category', 'filename': 'risk_by_charges.png'},
        {'title': 'Churn Probability Distribution', 'filename': 'risk_probability_histogram.png'},
    ]
    risk_charts = [c for c in risk_charts if os.path.exists(f'reports/figures/{c["filename"]}')]

    return render_template('risk.html', active_page='risk',
                           risk_stats=risk_stats, profile=profile,
                           high_risk_customers=high_risk_customers,
                           risk_charts=risk_charts)


@app.route('/recommendations')
def recommendations():
    recs = [
        {
            'number': 1,
            'title': 'Contract Migration Program',
            'icon': 'fa-file-contract',
            'finding': 'Month-to-month contracts have the highest churn rate at 42.6%.',
            'impact': '3,853 customers affected',
            'action': 'Offer incentives (e.g., first month free or discounts) to switch to 1-year or 2-year contracts.',
            'benefit': 'Increase customer commitment and reduce short-term churn.'
        },
        {
            'number': 2,
            'title': 'New Customer Onboarding',
            'icon': 'fa-user-plus',
            'finding': 'New customers (0-1 year tenure) have the highest churn rate at 47.4%.',
            'impact': '2,164 customers affected',
            'action': 'Implement targeted onboarding programs with early touchpoints, welcome calls, and dedicated support.',
            'benefit': 'Improve early retention and user experience during the critical first year.'
        },
        {
            'number': 3,
            'title': 'Auto-Pay Migration',
            'icon': 'fa-credit-card',
            'finding': 'Electronic check users have the highest churn rate at 45.1%.',
            'impact': '2,359 customers affected',
            'action': 'Encourage migration to automatic payments (credit card/bank transfer) via small discounts.',
            'benefit': 'Reduce payment friction and involuntary churn from missed payments.'
        },
        {
            'number': 4,
            'title': 'Tech Support Bundle',
            'icon': 'fa-headset',
            'finding': 'Customers without Tech Support churn at 41.5%.',
            'impact': '3,465 customers affected',
            'action': 'Bundle Tech Support with core internet services or offer a free 3-month trial.',
            'benefit': 'Increase product stickiness and customer satisfaction.'
        },
        {
            'number': 5,
            'title': 'Fiber Optic Service Review',
            'icon': 'fa-wifi',
            'finding': 'Fiber optic customers have the highest churn rate at 41.8%.',
            'impact': '3,090 customers affected',
            'action': 'Investigate Fiber Optic service quality, reliability, and competitive pricing.',
            'benefit': 'Address specific service issues and retain high-value broadband customers.'
        },
        {
            'number': 6,
            'title': 'Pricing & Plan Review',
            'icon': 'fa-tags',
            'finding': 'Churned customers pay $74.60 avg vs $61.34 for retained — a $13.26 gap.',
            'impact': 'All 7,021 customers potentially affected',
            'action': 'Conduct a pricing review and offer flexible, personalized plans for price-sensitive segments.',
            'benefit': 'Prevent cost-driven churn and improve perceived value.'
        },
    ]
    return render_template('recommendations.html', active_page='recommendations',
                           recommendations=recs)


# ── API ROUTES ────────────────────────────────────────────────

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Single customer prediction via JSON."""
    try:
        data = request.get_json(force=True)
        p = get_predictor()
        result = p.predict_single(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bulk', methods=['POST'])
def api_bulk():
    """Bulk prediction from CSV upload."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'}), 400

        # Read CSV
        df = pd.read_csv(file)

        # Predict
        p = get_predictor()
        result_df = p.predict_bulk(df)

        # Sort by risk (highest first)
        result_df = result_df.sort_values('Churn_Probability', ascending=False)

        # Stats
        total = len(result_df)
        high = len(result_df[result_df['Risk_Category'] == 'High'])
        medium = len(result_df[result_df['Risk_Category'] == 'Medium'])
        low = len(result_df[result_df['Risk_Category'] == 'Low'])

        # CSV for download
        csv_buf = io.StringIO()
        result_df.to_csv(csv_buf, index=False)
        csv_data = csv_buf.getvalue()

        # Row data for table (send all, JS limits display)
        display_cols = ['customerID', 'tenure', 'Contract', 'MonthlyCharges',
                        'InternetService', 'Churn_Probability', 'Risk_Category']
        avail = [c for c in display_cols if c in result_df.columns]
        rows = result_df[avail].head(200).to_dict('records')

        return jsonify({
            'total': total,
            'high_risk': high,
            'medium_risk': medium,
            'low_risk': low,
            'rows': rows,
            'csv_data': csv_data,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Run ───────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  ChurnIQ — Starting Web Application")
    print("=" * 50)
    print(f"  Model: {get_predictor().model_name}")
    print(f"  Open: http://localhost:5000")
    print("=" * 50 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
