import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Light pastel color palette from conventions
PALETTE = ['#A8D8EA', '#AA96DA', '#FCBAD3', '#FFFFD2', '#B5EAD7', '#FFB7B2', '#E2F0CB', '#C7CEEA']
sns.set_theme(style='whitegrid', palette=PALETTE)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['text.color'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'

def evaluate_model(y_true, y_pred, y_prob):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_prob)
    }

def main():
    print("=" * 50)
    print("Starting ML Prediction Module")
    print("=" * 50)
    
    # Define directories
    data_path = 'data/telco_churn_cleaned.csv'
    figures_dir = 'reports/figures/'
    models_dir = 'models/'
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load data
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return None, None, None
    df = pd.read_csv(data_path, encoding='utf-8')
    
    # 2. Prepare features
    if 'Churn' not in df.columns:
        print("Error: 'Churn' column not found in data.")
        return None, None, None
        
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    if df['Churn'].isnull().any():
        # Handle cases where Churn might be already 1/0 or missing
        df.dropna(subset=['Churn'], inplace=True)
        
    X = df.drop('Churn', axis=1)
    y = df['Churn'].astype(int)
    
    # Separate numeric and categorical
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    # If TotalCharges is not numeric (e.g. read as string), convert it
    for col in numeric_cols:
        if col in X.columns and X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col].replace(' ', np.nan))
    
    # Fill missing numeric with median just in case
    for col in numeric_cols:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())
            
    # One-hot encode all categorical columns
    categorical_cols = [c for c in X.columns if c not in numeric_cols and c != 'customerID']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Drop customerID if present
    if 'customerID' in X.columns:
        X = X.drop('customerID', axis=1)
        
    feature_columns = X.columns.tolist()
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # StandardScaler on numeric features
    scaler = StandardScaler()
    # Find which numeric cols actually survived in X
    num_cols_present = [c for c in numeric_cols if c in X_train.columns]
    if num_cols_present:
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[num_cols_present] = scaler.fit_transform(X_train[num_cols_present])
        X_test_scaled[num_cols_present] = scaler.transform(X_test[num_cols_present])
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
        
    # 4. Train models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    if XGB_AVAILABLE:
        models['XGBoost'] = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                                          random_state=42, eval_metric='logloss')
                                          
    # 5. Compute metrics
    results = {}
    best_model_name = None
    best_auc = -1
    best_model = None
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = y_pred
            
        metrics = evaluate_model(y_test, y_pred, y_prob)
        results[name] = metrics
        
        if metrics['AUC-ROC'] > best_auc:
            best_auc = metrics['AUC-ROC']
            best_model_name = name
            best_model = model
            
    # 6. Print comparison table
    print("\nModel Comparison:")
    results_df = pd.DataFrame(results).T
    print(results_df.round(4).to_string())
    
    # 7. Select best model
    print(f"\nBest Model selected based on AUC-ROC: {best_model_name} (AUC: {best_auc:.4f})")
    
    # 8. Generate charts
    # a. model_comparison.png
    fig, ax = plt.subplots(figsize=(10, 6))
    results_df.plot(kind='bar', ax=ax, color=PALETTE[:len(results_df.columns)])
    plt.title('Model Comparison Across Metrics', fontsize=16)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=15)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'model_comparison.png'))
    plt.close()
    
    # b. confusion_matrix.png
    num_models = len(models)
    rows = int(np.ceil(num_models / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4*rows))
    axes = axes.flatten()
    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        # Using a light blue/purple custom colormap
        cmap = sns.light_palette(PALETTE[1], as_cmap=True)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=axes[i], cbar=False)
        axes[i].set_title(f'Confusion Matrix: {name}', fontsize=12)
        axes[i].set_xlabel('Predicted', fontsize=10)
        axes[i].set_ylabel('Actual', fontsize=10)
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'confusion_matrix.png'))
    plt.close()
    
    # c. roc_curves.png
    plt.figure(figsize=(8, 6))
    for i, (name, model) in enumerate(models.items()):
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {results[name]["AUC-ROC"]:.3f})', color=PALETTE[i % len(PALETTE)], lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves', fontsize=16)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'roc_curves.png'))
    plt.close()
    
    # d. feature_importance.png
    importances = None
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_[0])
        
    if importances is not None:
        feat_imp_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(15)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette=PALETTE[:1])
        plt.title(f'Top 15 Feature Importances ({best_model_name})', fontsize=16)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'feature_importance.png'))
        plt.close()
    else:
        print("Model does not have feature importances or coefficients.")
        
    # 9. Save best model, scaler, and features
    model_save_path = os.path.join(models_dir, 'churn_model.pkl')
    print(f"\nSaving best model to {model_save_path}...")
    save_dict = {
        'model': best_model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'model_name': best_model_name
    }
    joblib.dump(save_dict, model_save_path)
    print("=" * 50)
    print("ML Prediction Module Completed")
    print("=" * 50)
    
    return best_model, scaler, results

if __name__ == '__main__':
    main()
