"""
Customer Churn Prediction — Main Pipeline
==========================================
Runs the complete churn analysis pipeline end-to-end.

Usage:
    python main.py

Steps:
    1. Data Cleaning
    2. Exploratory Data Analysis (EDA)
    3. Churn Pattern Analysis
    4. Customer Segmentation
    5. ML Churn Prediction
    6. Risk Analysis
    7. Business Recommendations
    8. Generate Dashboard
"""

import os
import sys
import time

# Fix Windows console encoding for emoji/Unicode characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

def run_step(step_num, step_name, module_name):
    """Run a pipeline step and handle errors."""
    print("\n" + "=" * 60)
    print(f"  STEP {step_num}: {step_name}")
    print("=" * 60)
    
    start = time.time()
    try:
        module = __import__(f'src.{module_name}', fromlist=['main'])
        result = module.main()
        elapsed = time.time() - start
        print(f"\n✓ Step {step_num} completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n✗ Step {step_num} failed after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_dashboard():
    """Generate the HTML dashboard after all analysis is complete."""
    print("\n" + "=" * 60)
    print("  STEP 8: GENERATING DASHBOARD")
    print("=" * 60)
    
    import base64
    import pandas as pd
    
    os.makedirs('dashboard', exist_ok=True)
    
    # Load risk scores for KPI calculations
    try:
        df = pd.read_csv('data/customer_risk_scores.csv')
        total_customers = len(df)
        churned = int(df['Churn'].sum())
        churn_rate = (churned / total_customers * 100)
        avg_monthly = df['MonthlyCharges'].mean()
        high_risk = len(df[df['Risk_Category'] == 'High Risk'])
        medium_risk = len(df[df['Risk_Category'] == 'Medium Risk'])
        low_risk = len(df[df['Risk_Category'] == 'Low Risk'])
    except Exception as e:
        print(f"Warning: Could not load risk scores: {e}")
        total_customers = churned = 0
        churn_rate = avg_monthly = 0
        high_risk = medium_risk = low_risk = 0
    
    # Load recommendations
    recommendations_html = ""
    try:
        with open('reports/recommendations.txt', 'r', encoding='utf-8') as f:
            recs = f.read()
        for line in recs.split('\n'):
            if line.startswith('Recommendation #'):
                recommendations_html += f'<h4 style="color:#5B6ABF; margin-top:18px;">{line}</h4>'
            elif line.startswith('- Finding:'):
                recommendations_html += f'<p><strong>📊 Finding:</strong> {line[10:]}</p>'
            elif line.startswith('- Impact:'):
                recommendations_html += f'<p><strong>👥 Impact:</strong> {line[9:]}</p>'
            elif line.startswith('- Recommendation:'):
                recommendations_html += f'<p><strong>💡 Action:</strong> {line[17:]}</p>'
            elif line.startswith('- Expected Benefit:'):
                recommendations_html += f'<p><strong>✅ Benefit:</strong> {line[19:]}</p>'
    except:
        recommendations_html = "<p>Recommendations not available. Run the full pipeline first.</p>"
    
    # Embed chart images as base64
    figures_dir = 'reports/figures'
    chart_sections = [
        ("Churn Distribution", "churn_distribution.png"),
        ("Churn by Contract Type", "churn_by_contract.png"),
        ("Churn by Tenure", "churn_by_tenure.png"),
        ("Churn by Monthly Charges", "churn_by_monthly_charges.png"),
        ("Churn by Payment Method", "churn_by_payment_method.png"),
        ("Churn by Internet Service", "churn_by_internet_service.png"),
        ("Churn by Tech Support", "churn_by_tech_support.png"),
        ("Churn by Gender", "churn_by_gender.png"),
        ("Churn by Senior Citizen", "churn_by_senior_citizen.png"),
        ("Churn by Partner & Dependents", "churn_by_dependents_partner.png"),
        ("Correlation Heatmap", "correlation_heatmap.png"),
        ("Top Churn Factors", "top_churn_factors.png"),
        ("Service Combination Churn", "service_combination_churn.png"),
        ("Churn by Tenure Segment", "segment_tenure_churn.png"),
        ("Churn by Charges Segment", "segment_charges_churn.png"),
        ("Segment Cross Analysis", "segment_cross_analysis.png"),
        ("Model Comparison", "model_comparison.png"),
        ("Confusion Matrices", "confusion_matrix.png"),
        ("ROC Curves", "roc_curves.png"),
        ("Feature Importance", "feature_importance.png"),
        ("Risk Distribution", "risk_distribution.png"),
        ("Risk by Tenure", "risk_by_tenure.png"),
        ("Risk by Monthly Charges", "risk_by_charges.png"),
        ("Churn Probability Distribution", "risk_probability_histogram.png"),
    ]
    
    charts_html = ""
    for title, filename in chart_sections:
        filepath = os.path.join(figures_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode('utf-8')
            charts_html += f'''
            <div class="chart-card">
                <h3>{title}</h3>
                <img src="data:image/png;base64,{img_b64}" alt="{title}">
            </div>
            '''
    
    # High-risk customers table
    risk_table_html = ""
    try:
        high_risk_df = df.nlargest(10, 'Churn_Probability')
        cols_to_show = ['customerID', 'tenure', 'Contract', 'MonthlyCharges', 'InternetService', 'TechSupport', 'Churn_Probability', 'Risk_Category']
        cols_available = [c for c in cols_to_show if c in high_risk_df.columns]
        
        risk_table_html = "<table><thead><tr>"
        for col in cols_available:
            risk_table_html += f"<th>{col}</th>"
        risk_table_html += "</tr></thead><tbody>"
        for _, row in high_risk_df[cols_available].iterrows():
            risk_table_html += "<tr>"
            for col in cols_available:
                val = row[col]
                if col == 'Churn_Probability':
                    val = f"{val:.1%}"
                elif col == 'Risk_Category':
                    color = '#FF6B6B' if val == 'High Risk' else '#FFD93D' if val == 'Medium Risk' else '#6BCB77'
                    val = f'<span style="color:{color}; font-weight:bold;">{val}</span>'
                risk_table_html += f"<td>{val}</td>"
            risk_table_html += "</tr>"
        risk_table_html += "</tbody></table>"
    except:
        risk_table_html = "<p>Risk data not available.</p>"
    
    # Build the full HTML dashboard
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Churn Analytics Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #F8F9FE 0%, #EEF1F8 100%);
            color: #333;
            line-height: 1.6;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 40px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        }}
        
        .header h1 {{
            font-size: 2.2em;
            font-weight: 300;
            letter-spacing: 1px;
        }}
        
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
            margin-top: 8px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 30px 20px;
        }}
        
        /* KPI Cards */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .kpi-card {{
            background: white;
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .kpi-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }}
        
        .kpi-card .kpi-icon {{
            font-size: 2em;
            margin-bottom: 8px;
        }}
        
        .kpi-card .kpi-value {{
            font-size: 2em;
            font-weight: 700;
            color: #5B6ABF;
        }}
        
        .kpi-card .kpi-label {{
            font-size: 0.9em;
            color: #888;
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .kpi-card.danger .kpi-value {{ color: #FF6B6B; }}
        .kpi-card.warning .kpi-value {{ color: #FFB347; }}
        .kpi-card.success .kpi-value {{ color: #6BCB77; }}
        
        /* Section Headers */
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            font-size: 1.6em;
            color: #5B6ABF;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #E8E8F4;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        /* Chart Grid */
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 24px;
        }}
        
        .chart-card {{
            background: white;
            border-radius: 16px;
            padding: 20px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        }}
        
        .chart-card h3 {{
            color: #5B6ABF;
            margin-bottom: 12px;
            font-size: 1.1em;
        }}
        
        .chart-card img {{
            width: 100%;
            height: auto;
            border-radius: 8px;
        }}
        
        /* Table */
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 14px 16px;
            text-align: left;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        td {{
            padding: 12px 16px;
            border-bottom: 1px solid #F0F0F5;
        }}
        
        tr:hover td {{
            background: #F8F9FE;
        }}
        
        /* Recommendations */
        .rec-card {{
            background: white;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
            margin-bottom: 16px;
        }}
        
        .rec-card h4 {{
            color: #5B6ABF;
        }}
        
        .rec-card p {{
            margin: 6px 0;
            color: #555;
        }}
        
        /* Risk Badges */
        .risk-summary {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        
        .risk-badge {{
            text-align: center;
            padding: 20px;
            border-radius: 12px;
            color: white;
            font-weight: 600;
        }}
        
        .risk-badge.high {{ background: linear-gradient(135deg, #FF6B6B, #ee5a5a); }}
        .risk-badge.medium {{ background: linear-gradient(135deg, #FFD93D, #f0c929); }}
        .risk-badge.low {{ background: linear-gradient(135deg, #6BCB77, #5ab868); }}
        
        .risk-badge .badge-count {{
            font-size: 2em;
            display: block;
        }}
        
        .risk-badge .badge-label {{
            font-size: 0.85em;
            opacity: 0.9;
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 30px;
            color: #aaa;
            font-size: 0.85em;
        }}
        
        /* Nav Tabs */
        .nav {{
            display: flex;
            gap: 8px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        
        .nav button {{
            padding: 10px 20px;
            border: none;
            background: white;
            color: #5B6ABF;
            border-radius: 25px;
            cursor: pointer;
            font-size: 0.95em;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            transition: all 0.2s;
        }}
        
        .nav button:hover, .nav button.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }}
        
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        
        @media (max-width: 768px) {{
            .chart-grid {{ grid-template-columns: 1fr; }}
            .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .risk-summary {{ grid-template-columns: 1fr; }}
            .header h1 {{ font-size: 1.5em; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Customer Churn Analytics Dashboard</h1>
        <p>Telecom Customer Churn Prediction & Business Intelligence</p>
    </div>
    
    <div class="container">
        <!-- Navigation -->
        <div class="nav">
            <button class="active" onclick="showTab('overview')">📈 Overview</button>
            <button onclick="showTab('eda')">🔍 EDA Analysis</button>
            <button onclick="showTab('ml')">🤖 ML Models</button>
            <button onclick="showTab('risk')">⚠️ Risk Analysis</button>
            <button onclick="showTab('recs')">💡 Recommendations</button>
        </div>
        
        <!-- Overview Tab -->
        <div id="overview" class="tab-content active">
            <div class="section">
                <h2>📊 Key Performance Indicators</h2>
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-icon">👥</div>
                        <div class="kpi-value">{total_customers:,}</div>
                        <div class="kpi-label">Total Customers</div>
                    </div>
                    <div class="kpi-card danger">
                        <div class="kpi-icon">🚪</div>
                        <div class="kpi-value">{churned:,}</div>
                        <div class="kpi-label">Churned Customers</div>
                    </div>
                    <div class="kpi-card warning">
                        <div class="kpi-icon">📉</div>
                        <div class="kpi-value">{churn_rate:.1f}%</div>
                        <div class="kpi-label">Churn Rate</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-icon">💰</div>
                        <div class="kpi-value">${avg_monthly:.0f}</div>
                        <div class="kpi-label">Avg Monthly Charges</div>
                    </div>
                    <div class="kpi-card danger">
                        <div class="kpi-icon">🔴</div>
                        <div class="kpi-value">{high_risk:,}</div>
                        <div class="kpi-label">High Risk Customers</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>⚠️ Risk Summary</h2>
                <div class="risk-summary">
                    <div class="risk-badge high">
                        <span class="badge-count">{high_risk:,}</span>
                        <span class="badge-label">🔴 High Risk (60-100%)</span>
                    </div>
                    <div class="risk-badge medium">
                        <span class="badge-count">{medium_risk:,}</span>
                        <span class="badge-label">🟡 Medium Risk (30-60%)</span>
                    </div>
                    <div class="risk-badge low">
                        <span class="badge-count">{low_risk:,}</span>
                        <span class="badge-label">🟢 Low Risk (0-30%)</span>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🔴 Top 10 High-Risk Customers</h2>
                {risk_table_html}
            </div>
        </div>
        
        <!-- EDA Tab -->
        <div id="eda" class="tab-content">
            <div class="section">
                <h2>🔍 Exploratory Data Analysis</h2>
                <div class="chart-grid">
                    {charts_html}
                </div>
            </div>
        </div>
        
        <!-- ML Tab -->
        <div id="ml" class="tab-content">
            <div class="section">
                <h2>🤖 Machine Learning Model Results</h2>
                <div class="chart-grid">
                </div>
            </div>
        </div>
        
        <!-- Risk Tab -->
        <div id="risk" class="tab-content">
            <div class="section">
                <h2>⚠️ Customer Risk Analysis</h2>
                <div class="chart-grid">
                </div>
            </div>
        </div>
        
        <!-- Recommendations Tab -->
        <div id="recs" class="tab-content">
            <div class="section">
                <h2>💡 Data-Driven Retention Recommendations</h2>
                <div class="rec-card">
                    {recommendations_html}
                </div>
            </div>
        </div>
    </div>
    
    <div class="footer">
        <p>Customer Churn Analytics Dashboard | Built with Python, Scikit-learn & XGBoost</p>
    </div>
    
    <script>
        function showTab(tabId) {{
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.nav button').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }}
        
        // Distribute charts to appropriate tabs
        document.addEventListener('DOMContentLoaded', function() {{
            const edaGrid = document.querySelector('#eda .chart-grid');
            const mlGrid = document.querySelector('#ml .chart-grid');
            const riskGrid = document.querySelector('#risk .chart-grid');
            
            const allCharts = document.querySelectorAll('#eda .chart-card');
            
            const mlKeywords = ['Model Comparison', 'Confusion', 'ROC Curves', 'Feature Importance'];
            const riskKeywords = ['Risk Distribution', 'Risk by Tenure', 'Risk by Monthly', 'Churn Probability Distribution'];
            
            allCharts.forEach(card => {{
                const title = card.querySelector('h3').textContent;
                let moved = false;
                
                for (const kw of mlKeywords) {{
                    if (title.includes(kw)) {{
                        mlGrid.appendChild(card.cloneNode(true));
                        card.style.display = 'none';
                        moved = true;
                        break;
                    }}
                }}
                
                if (!moved) {{
                    for (const kw of riskKeywords) {{
                        if (title.includes(kw)) {{
                            riskGrid.appendChild(card.cloneNode(true));
                            card.style.display = 'none';
                            break;
                        }}
                    }}
                }}
            }});
        }});
    </script>
</body>
</html>'''
    
    dashboard_path = 'dashboard/churn_dashboard.html'
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n✓ Dashboard saved to {dashboard_path}")
    print("  Open this file in a browser to view the dashboard.")


def main():
    """Run the complete Customer Churn Prediction pipeline."""
    start_time = time.time()
    
    print("\n" + "🔶" * 30)
    print("  CUSTOMER CHURN PREDICTION — FULL PIPELINE")
    print("🔶" * 30)
    
    # Step 1: Data Cleaning
    cleaned_df = run_step(1, "DATA CLEANING", "data_cleaning")
    if cleaned_df is None:
        print("\n❌ Pipeline stopped: Data cleaning failed.")
        return
    
    # Step 2: EDA Analysis
    run_step(2, "EXPLORATORY DATA ANALYSIS", "eda_analysis")
    
    # Step 3: Churn Pattern Analysis
    run_step(3, "CHURN PATTERN ANALYSIS", "churn_patterns")
    
    # Step 4: Customer Segmentation
    run_step(4, "CUSTOMER SEGMENTATION", "customer_segmentation")
    
    # Step 5: ML Prediction
    ml_result = run_step(5, "ML CHURN PREDICTION", "ml_prediction")
    if ml_result is None:
        print("\n❌ Pipeline stopped: ML training failed.")
        return
    
    # Step 6: Risk Analysis
    run_step(6, "RISK ANALYSIS", "risk_analysis")
    
    # Step 7: Business Recommendations
    run_step(7, "BUSINESS RECOMMENDATIONS", "recommendations")
    
    # Step 8: Generate Dashboard
    generate_dashboard()
    
    # Final Summary
    elapsed = time.time() - start_time
    print("\n" + "✅" * 30)
    print("  PIPELINE COMPLETE!")
    print("✅" * 30)
    print(f"\n  Total time: {elapsed:.1f} seconds")
    print("\n  Generated outputs:")
    print("  📁 data/telco_churn_cleaned.csv       — Cleaned dataset")
    print("  📁 data/customer_risk_scores.csv       — Risk-scored customers")
    print("  📁 reports/figures/                     — All analysis charts")
    print("  📁 reports/recommendations.txt         — Business recommendations")
    print("  📁 models/churn_model.pkl              — Trained ML model")
    print("  📁 dashboard/churn_dashboard.html      — Interactive dashboard")
    print("\n  🌐 Open dashboard/churn_dashboard.html in your browser!")


if __name__ == "__main__":
    main()
