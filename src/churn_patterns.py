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
    print("CHURN PATTERN ANALYSIS")
    print("=" * 50)
    
    os.makedirs('reports/figures', exist_ok=True)
    
    df = pd.read_csv('data/telco_churn_cleaned.csv')
    df['Churn_Bin'] = (df['Churn'] == 'Yes').astype(int)
    
    categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                        'PhoneService', 'MultipleLines', 'InternetService', 
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 
                        'Contract', 'PaperlessBilling', 'PaymentMethod']
                        
    churn_rates = []
    for col in categorical_cols:
        if col in df.columns:
            grouped = df.groupby(col)['Churn_Bin'].agg(['mean', 'count']).reset_index()
            grouped.columns = ['Value', 'ChurnRate', 'Count']
            grouped['Feature'] = col
            # Standardize Value to string for combining
            grouped['Value'] = grouped['Value'].astype(str)
            churn_rates.append(grouped)
            
    churn_rates_df = pd.concat(churn_rates, ignore_index=True)
    churn_rates_df = churn_rates_df[churn_rates_df['Count'] >= 100] # Filter small groups
    churn_rates_df = churn_rates_df.sort_values('ChurnRate', ascending=False)
    
    print("\n--- Churn Rate by Feature/Value ---")
    print(churn_rates_df.head(20).to_string(index=False))
    
    top_5 = churn_rates_df.head(5)
    
    # Analyze service combinations
    combos = []
    
    # 1. Fiber optic + No OnlineSecurity + No TechSupport
    if 'InternetService' in df.columns and 'OnlineSecurity' in df.columns and 'TechSupport' in df.columns:
        mask1 = (df['InternetService'] == 'Fiber optic') & (df['OnlineSecurity'] == 'No') & (df['TechSupport'] == 'No')
        if mask1.sum() > 0:
            rate1 = df[mask1]['Churn_Bin'].mean()
            combos.append({'Combo': 'Fiber + No Security/Tech', 'ChurnRate': rate1, 'Count': mask1.sum()})
    
    # 2. Month-to-month + Electronic check
    if 'Contract' in df.columns and 'PaymentMethod' in df.columns:
        mask2 = (df['Contract'] == 'Month-to-month') & (df['PaymentMethod'] == 'Electronic check')
        if mask2.sum() > 0:
            rate2 = df[mask2]['Churn_Bin'].mean()
            combos.append({'Combo': 'Month-to-month + Elec Check', 'ChurnRate': rate2, 'Count': mask2.sum()})
    
    # 3. No Partner + No Dependents
    if 'Partner' in df.columns and 'Dependents' in df.columns:
        mask3 = (df['Partner'] == 'No') & (df['Dependents'] == 'No')
        if mask3.sum() > 0:
            rate3 = df[mask3]['Churn_Bin'].mean()
            combos.append({'Combo': 'Single + No Dependents', 'ChurnRate': rate3, 'Count': mask3.sum()})
    
    combos_df = pd.DataFrame(combos).sort_values('ChurnRate', ascending=False)
    
    print("\n--- Service Combinations Churn Rate ---")
    print(combos_df.to_string(index=False))
    
    # Charts
    sns.set_style('whitegrid')
    palette = ['#A8D8EA', '#AA96DA', '#FCBAD3', '#FFFFD2', '#B5EAD7', '#FFB7B2', '#E2F0CB', '#C7CEEA']
    
    plt.figure(figsize=(10, 6))
    top_10 = churn_rates_df.head(10).copy()
    top_10['Feature_Value'] = top_10['Feature'] + ': ' + top_10['Value']
    sns.barplot(data=top_10, x='ChurnRate', y='Feature_Value', palette=palette)
    plt.title('Top 10 Churn Factors', color='#333333')
    plt.xlabel('Churn Rate', color='#333333')
    plt.ylabel('Feature: Value', color='#333333')
    plt.tight_layout()
    plt.savefig('reports/figures/top_churn_factors.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=combos_df, x='ChurnRate', y='Combo', palette=palette)
    plt.title('Churn Rates for Service Combinations', color='#333333')
    plt.xlabel('Churn Rate', color='#333333')
    plt.ylabel('Combination', color='#333333')
    plt.tight_layout()
    plt.savefig('reports/figures/service_combination_churn.png')
    plt.close()
    
    print("\n" + "=" * 50)
    print("KEY FINDINGS")
    print("=" * 50)
    print("1. Highest churn driver is:", top_5.iloc[0]['Feature'], "with value", top_5.iloc[0]['Value'], f"({top_5.iloc[0]['ChurnRate']:.2%} churn)")
    print("2. Second highest driver:", top_5.iloc[1]['Feature'], "with value", top_5.iloc[1]['Value'], f"({top_5.iloc[1]['ChurnRate']:.2%} churn)")
    
    combo_1 = combos_df[combos_df['Combo']=='Month-to-month + Elec Check']
    if not combo_1.empty:
        print(f"3. High risk combination (Month-to-month + Elec check) has churn rate of {combo_1['ChurnRate'].values[0]:.2%}")
        
    combo_2 = combos_df[combos_df['Combo']=='Fiber + No Security/Tech']
    if not combo_2.empty:
        print(f"4. Fiber optic customers without Security/Tech support are highly vulnerable: {combo_2['ChurnRate'].values[0]:.2%} churn")
        
    print("5. Month-to-month contracts strongly correlate with other high-churn indicators.")
    
    return {
        'top_factors': top_5.to_dict('records'),
        'combos': combos_df.to_dict('records')
    }

if __name__ == '__main__':
    main()
