import pandas as pd
import os

def main():
    print("=" * 50)
    print("Starting Recommendations Module")
    print("=" * 50)
    
    os.makedirs('reports', exist_ok=True)
    
    # 1. Load data
    df = pd.read_csv('data/customer_risk_scores.csv')
    
    recommendations = []
    
    # 2. Analyze data
    
    # Contract type
    if 'Contract' in df.columns:
        cr_by_contract = df.groupby('Contract')['Churn'].mean() * 100
        if 'Month-to-month' in cr_by_contract and cr_by_contract['Month-to-month'] == cr_by_contract.max():
            impact = len(df[df['Contract'] == 'Month-to-month'])
            recommendations.append({
                'Finding': f"Month-to-month contracts have the highest churn rate at {cr_by_contract['Month-to-month']:.1f}%.",
                'Impact': f"{impact} customers affected.",
                'Recommendation': "Offer incentives (e.g., first month free or discounts) to switch to 1-year or 2-year contracts.",
                'Expected Benefit': "Increase customer commitment and reduce short-term churn."
            })
            
    # Tenure group
    if 'tenure' in df.columns:
        df['Tenure_Group'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 48, 60, 100], labels=['0-1 yr', '1-2 yrs', '2-4 yrs', '4-5 yrs', '5+ yrs'])
        cr_by_tenure = df.groupby('Tenure_Group', observed=True)['Churn'].mean() * 100
        if cr_by_tenure.idxmax() == '0-1 yr':
            impact = len(df[df['Tenure_Group'] == '0-1 yr'])
            recommendations.append({
                'Finding': f"New customers (0-1 yr) have the highest churn rate at {cr_by_tenure['0-1 yr']:.1f}%.",
                'Impact': f"{impact} customers affected.",
                'Recommendation': "Implement targeted onboarding programs and early touchpoints.",
                'Expected Benefit': "Improve early retention and user experience."
            })
            
    # Payment method
    if 'PaymentMethod' in df.columns:
        cr_by_payment = df.groupby('PaymentMethod')['Churn'].mean() * 100
        max_payment = cr_by_payment.idxmax()
        if 'Electronic check' in max_payment:
            impact = len(df[df['PaymentMethod'] == max_payment])
            recommendations.append({
                'Finding': f"Electronic check users have the highest churn rate at {cr_by_payment[max_payment]:.1f}%.",
                'Impact': f"{impact} customers affected.",
                'Recommendation': "Encourage migration to automatic payments (Credit Card/Bank Transfer) via small discounts.",
                'Expected Benefit': "Reduce payment friction and involuntary churn."
            })
            
    # Tech Support
    if 'TechSupport' in df.columns:
        cr_by_support = df.groupby('TechSupport')['Churn'].mean() * 100
        if 'No' in cr_by_support and cr_by_support['No'] == cr_by_support.max():
            impact = len(df[df['TechSupport'] == 'No'])
            recommendations.append({
                'Finding': f"Customers without Tech Support churn at a rate of {cr_by_support['No']:.1f}%.",
                'Impact': f"{impact} customers affected.",
                'Recommendation': "Bundle Tech Support with core internet services or offer a free trial.",
                'Expected Benefit': "Increase product stickiness and customer satisfaction."
            })
            
    # Internet Service
    if 'InternetService' in df.columns:
        cr_by_internet = df.groupby('InternetService')['Churn'].mean() * 100
        max_internet = cr_by_internet.idxmax()
        if 'Fiber optic' in max_internet:
            impact = len(df[df['InternetService'] == max_internet])
            recommendations.append({
                'Finding': f"Fiber optic customers have the highest churn rate at {cr_by_internet[max_internet]:.1f}%.",
                'Impact': f"{impact} customers affected.",
                'Recommendation': "Investigate Fiber Optic service quality, reliability, and pricing.",
                'Expected Benefit': "Address specific service issues and retain high-value customers."
            })
            
    # Monthly Charges
    if 'MonthlyCharges' in df.columns:
        avg_charge_churn = df[df['Churn'] == 1]['MonthlyCharges'].mean()
        avg_charge_stay = df[df['Churn'] == 0]['MonthlyCharges'].mean()
        if avg_charge_churn > avg_charge_stay:
            recommendations.append({
                'Finding': f"Churned customers pay on average ${avg_charge_churn:.2f}, which is higher than retained customers (${avg_charge_stay:.2f}).",
                'Impact': f"All {len(df)} customers potentially impacted by pricing.",
                'Recommendation': "Conduct a pricing review and offer flexible, personalized plans.",
                'Expected Benefit': "Prevent customers from leaving due to cost concerns."
            })
            
    # 3. Format & 4. Print & 5. Save
    with open('reports/recommendations.txt', 'w', encoding='utf-8') as f:
        f.write("DATA-DRIVEN BUSINESS RECOMMENDATIONS\n")
        f.write("========================================\n\n")
        
        for i, rec in enumerate(recommendations, 1):
            report_text = (
                f"Recommendation #{i}:\n"
                f"- Finding: {rec['Finding']}\n"
                f"- Impact: {rec['Impact']}\n"
                f"- Recommendation: {rec['Recommendation']}\n"
                f"- Expected Benefit: {rec['Expected Benefit']}\n\n"
            )
            print(report_text, end='')
            f.write(report_text)
            
    print("=" * 50)
    return recommendations

if __name__ == '__main__':
    main()
