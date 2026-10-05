import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def main():
    print("=" * 50)
    print("STARTING EDA ANALYSIS")
    print("=" * 50)
    
    data_path = 'data/telco_churn_cleaned.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run data_cleaning.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    # Create figures directory
    figures_dir = 'reports/figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Styling
    custom_palette = ['#A8D8EA', '#AA96DA', '#FCBAD3', '#FFFFD2', '#B5EAD7', '#FFB7B2', '#E2F0CB', '#C7CEEA']
    sns.set_style('whitegrid')
    sns.set_palette(custom_palette)
    
    def add_bar_labels(ax, total_count=None):
        for p in ax.patches:
            height = p.get_height()
            if np.isnan(height) or height == 0:
                continue
            ax.annotate(f'{height:g}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=10, color='#333333', xytext=(0, 5),
                        textcoords='offset points')

    print("\n" + "=" * 50)
    print("GENERATING CHARTS")
    print("=" * 50)
    
    # a. churn_distribution.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    churn_counts = df['Churn'].value_counts()
    ax1.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', colors=[custom_palette[0], custom_palette[5]], startangle=90)
    ax1.set_title('Churn Distribution', color='#333333')
    sns.countplot(data=df, x='Churn', ax=ax2, order=['No', 'Yes'], palette=[custom_palette[0], custom_palette[5]])
    ax2.set_title('Churn Counts', color='#333333')
    add_bar_labels(ax2)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'churn_distribution.png'), dpi=300)
    plt.close()
    print("Saved churn_distribution.png")
    
    # b. churn_by_contract.png
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='Contract', hue='Churn', palette=[custom_palette[0], custom_palette[5]])
    plt.title('Churn by Contract Type', color='#333333')
    add_bar_labels(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'churn_by_contract.png'), dpi=300)
    plt.close()
    print("Saved churn_by_contract.png")
    
    # c. churn_by_tenure.png
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='tenure', hue='Churn', kde=True, palette=[custom_palette[0], custom_palette[5]], element="step")
    plt.title('Tenure Distribution by Churn Status', color='#333333')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'churn_by_tenure.png'), dpi=300)
    plt.close()
    print("Saved churn_by_tenure.png")
    
    # d. churn_by_monthly_charges.png
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette=[custom_palette[0], custom_palette[5]])
    plt.title('Monthly Charges by Churn Status', color='#333333')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'churn_by_monthly_charges.png'), dpi=300)
    plt.close()
    print("Saved churn_by_monthly_charges.png")
    
    # e. churn_by_payment_method.png
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='PaymentMethod', hue='Churn', palette=[custom_palette[0], custom_palette[5]])
    plt.title('Churn by Payment Method', color='#333333')
    plt.xticks(rotation=15)
    add_bar_labels(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'churn_by_payment_method.png'), dpi=300)
    plt.close()
    print("Saved churn_by_payment_method.png")
    
    # f. churn_by_tech_support.png
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='TechSupport', hue='Churn', palette=[custom_palette[0], custom_palette[5]])
    plt.title('Churn by Tech Support', color='#333333')
    add_bar_labels(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'churn_by_tech_support.png'), dpi=300)
    plt.close()
    print("Saved churn_by_tech_support.png")
    
    # g. churn_by_internet_service.png
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='InternetService', hue='Churn', palette=[custom_palette[0], custom_palette[5]])
    plt.title('Churn by Internet Service', color='#333333')
    add_bar_labels(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'churn_by_internet_service.png'), dpi=300)
    plt.close()
    print("Saved churn_by_internet_service.png")
    
    # h. churn_by_gender.png
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='gender', hue='Churn', palette=[custom_palette[0], custom_palette[5]])
    plt.title('Churn by Gender', color='#333333')
    add_bar_labels(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'churn_by_gender.png'), dpi=300)
    plt.close()
    print("Saved churn_by_gender.png")
    
    # i. churn_by_senior_citizen.png
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='SeniorCitizen', hue='Churn', palette=[custom_palette[0], custom_palette[5]])
    plt.title('Churn by Senior Citizen', color='#333333')
    add_bar_labels(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'churn_by_senior_citizen.png'), dpi=300)
    plt.close()
    print("Saved churn_by_senior_citizen.png")
    
    # j. correlation_heatmap.png
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap=sns.color_palette("light:b", as_cmap=True), fmt=".2f")
    plt.title('Correlation Heatmap of Numeric Features', color='#333333')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    print("Saved correlation_heatmap.png")
    
    # k. churn_by_dependents_partner.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.countplot(data=df, x='Partner', hue='Churn', ax=ax1, palette=[custom_palette[0], custom_palette[5]])
    ax1.set_title('Churn by Partner', color='#333333')
    add_bar_labels(ax1)
    
    sns.countplot(data=df, x='Dependents', hue='Churn', ax=ax2, palette=[custom_palette[0], custom_palette[5]])
    ax2.set_title('Churn by Dependents', color='#333333')
    add_bar_labels(ax2)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'churn_by_dependents_partner.png'), dpi=300)
    plt.close()
    print("Saved churn_by_dependents_partner.png")
    
    print("\n" + "=" * 50)
    print("EDA ANALYSIS COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()
