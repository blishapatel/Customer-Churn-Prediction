import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 50)
    print("CUSTOMER SEGMENTATION")
    print("=" * 50)
    
    os.makedirs('reports/figures', exist_ok=True)
    
    df = pd.read_csv('data/telco_churn_cleaned.csv')
    df['Churn_Bin'] = (df['Churn'] == 'Yes').astype(int)
    
    # Handle TotalCharges converting to numeric if it is string
    if df['TotalCharges'].dtype == 'object':
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', ''), errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # 2. Tenure groups
    bins = [-1, 12, 24, 48, 72, np.inf]
    labels = ['New (0-12)', 'Medium (13-24)', 'Established (25-48)', 'Loyal (49-72)', 'Veteran (73+)']
    df['TenureGroup'] = pd.cut(df['tenure'], bins=bins, labels=labels)
    # restrict to the requested 4 groups if maximum tenure is <=72
    if df['tenure'].max() <= 72:
        bins = [-1, 12, 24, 48, 72]
        labels = ['New (0-12)', 'Medium (13-24)', 'Established (25-48)', 'Loyal (49-72)']
        df['TenureGroup'] = pd.cut(df['tenure'], bins=bins, labels=labels)
    
    # 3. Charge groups
    df['ChargeGroup'] = pd.qcut(df['MonthlyCharges'], q=4, labels=['Low', 'Medium', 'High', 'Premium'])
    
    # 4. Profile tenure segments
    print("\n--- Tenure Segments Profile ---")
    tenure_profile = df.groupby('TenureGroup', observed=False).agg(
        Count=('tenure', 'count'),
        ChurnRate=('Churn_Bin', 'mean'),
        AvgMonthlyCharges=('MonthlyCharges', 'mean'),
        AvgTotalCharges=('TotalCharges', 'mean')
    ).reset_index()
    tenure_profile['MostCommonContract'] = df.groupby('TenureGroup', observed=False)['Contract'].agg(lambda x: x.mode()[0] if not x.empty else 'N/A').values
    tenure_profile['MostCommonInternet'] = df.groupby('TenureGroup', observed=False)['InternetService'].agg(lambda x: x.mode()[0] if not x.empty else 'N/A').values
    print(tenure_profile.to_string(index=False))
    
    # 5. Profile charge segments
    print("\n--- Charge Segments Profile ---")
    charge_profile = df.groupby('ChargeGroup', observed=False).agg(
        Count=('tenure', 'count'),
        ChurnRate=('Churn_Bin', 'mean'),
        AvgMonthlyCharges=('MonthlyCharges', 'mean'),
        AvgTotalCharges=('TotalCharges', 'mean')
    ).reset_index()
    charge_profile['MostCommonContract'] = df.groupby('ChargeGroup', observed=False)['Contract'].agg(lambda x: x.mode()[0] if not x.empty else 'N/A').values
    charge_profile['MostCommonInternet'] = df.groupby('ChargeGroup', observed=False)['InternetService'].agg(lambda x: x.mode()[0] if not x.empty else 'N/A').values
    print(charge_profile.to_string(index=False))
    
    # 6. Cross-segment analysis
    print("\n--- Cross-Segment Analysis (Tenure x Contract) Churn Rates ---")
    cross_analysis = df.pivot_table(index='TenureGroup', columns='Contract', values='Churn_Bin', aggfunc='mean', observed=False)
    print(cross_analysis)
    
    # 7. Charts
    sns.set_style('whitegrid')
    palette = ['#A8D8EA', '#AA96DA', '#FCBAD3', '#FFFFD2', '#B5EAD7', '#FFB7B2', '#E2F0CB', '#C7CEEA']
    
    # a. segment_tenure_churn.png
    plt.figure(figsize=(8, 5))
    sns.barplot(data=tenure_profile, x='TenureGroup', y='ChurnRate', palette=palette)
    plt.title('Churn Rate by Tenure Group', color='#333333')
    plt.ylabel('Churn Rate')
    plt.tight_layout()
    plt.savefig('reports/figures/segment_tenure_churn.png')
    plt.close()
    
    # b. segment_charges_churn.png
    plt.figure(figsize=(8, 5))
    sns.barplot(data=charge_profile, x='ChargeGroup', y='ChurnRate', palette=palette)
    plt.title('Churn Rate by Charge Group', color='#333333')
    plt.ylabel('Churn Rate')
    plt.tight_layout()
    plt.savefig('reports/figures/segment_charges_churn.png')
    plt.close()
    
    # c. segment_cross_analysis.png
    plt.figure(figsize=(8, 6))
    sns.heatmap(cross_analysis, annot=True, fmt=".2%", cmap=sns.color_palette("light:#AA96DA", as_cmap=True))
    plt.title('Churn Rate Heatmap: Tenure vs Contract', color='#333333')
    plt.tight_layout()
    plt.savefig('reports/figures/segment_cross_analysis.png')
    plt.close()
    
    # d. segment_profiles.png (Multi-panel)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=tenure_profile, x='TenureGroup', y='Count', palette=palette, ax=axes[0])
    axes[0].set_title('Customer Count by Tenure', color='#333333')
    sns.barplot(data=charge_profile, x='ChargeGroup', y='Count', palette=palette, ax=axes[1])
    axes[1].set_title('Customer Count by Charge Group', color='#333333')
    plt.tight_layout()
    plt.savefig('reports/figures/segment_profiles.png')
    plt.close()
    
    return df

if __name__ == '__main__':
    main()
