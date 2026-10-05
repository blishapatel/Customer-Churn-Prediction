"""
ChurnIQ — Prediction Engine
Wraps the trained ML model for single & bulk predictions with explanations.
"""

import os
import pandas as pd
import numpy as np
import joblib


class ChurnPredictor:
    """Loads the trained model and provides prediction + explanation methods."""

    FEATURE_LABELS = {
        'tenure': 'Tenure (months)',
        'MonthlyCharges': 'Monthly Charges',
        'TotalCharges': 'Total Charges',
        'Contract_One year': 'One Year Contract',
        'Contract_Two year': 'Two Year Contract',
        'Contract_Month-to-month': 'Month-to-month Contract',
        'InternetService_Fiber optic': 'Fiber Optic Internet',
        'InternetService_DSL': 'DSL Internet',
        'InternetService_No': 'No Internet Service',
        'PaymentMethod_Electronic check': 'Electronic Check Payment',
        'PaymentMethod_Mailed check': 'Mailed Check Payment',
        'PaymentMethod_Bank transfer (automatic)': 'Bank Transfer (Auto)',
        'PaymentMethod_Credit card (automatic)': 'Credit Card (Auto)',
        'TechSupport_Yes': 'Has Tech Support',
        'TechSupport_No': 'No Tech Support',
        'OnlineSecurity_Yes': 'Has Online Security',
        'OnlineSecurity_No': 'No Online Security',
        'PaperlessBilling_Yes': 'Paperless Billing',
        'OnlineBackup_Yes': 'Has Online Backup',
        'OnlineBackup_No': 'No Online Backup',
        'DeviceProtection_Yes': 'Has Device Protection',
        'DeviceProtection_No': 'No Device Protection',
        'StreamingTV_Yes': 'Has Streaming TV',
        'StreamingMovies_Yes': 'Has Streaming Movies',
        'Dependents_Yes': 'Has Dependents',
        'Partner_Yes': 'Has Partner',
        'SeniorCitizen_Yes': 'Senior Citizen',
        'gender_Male': 'Male',
        'PhoneService_Yes': 'Has Phone Service',
        'MultipleLines_Yes': 'Has Multiple Lines',
    }

    def __init__(self, model_path='models/churn_model.pkl'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run 'python main.py' first."
            )
        data = joblib.load(model_path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.model_name = data.get('model_name', 'Unknown')

    # ── helpers ───────────────────────────────────────────────
    @staticmethod
    def risk_category(prob):
        if prob <= 0.30:
            return 'Low'
        elif prob <= 0.60:
            return 'Medium'
        return 'High'

    def _prepare_row(self, row_dict):
        """Turn a flat customer dict into a model-ready 1-row DataFrame."""
        df = pd.DataFrame([row_dict])

        # Ensure numeric
        for col in ('tenure', 'MonthlyCharges', 'TotalCharges'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # If TotalCharges missing, estimate from tenure * MonthlyCharges
        if 'TotalCharges' not in df.columns or df['TotalCharges'].iloc[0] == 0:
            df['TotalCharges'] = df['tenure'] * df['MonthlyCharges']

        # One-hot encode
        cat_cols = [c for c in df.columns if c not in ('tenure', 'MonthlyCharges', 'TotalCharges')]
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        # Align with training columns
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns]

        # Scale numeric
        num_cols = [c for c in ('tenure', 'MonthlyCharges', 'TotalCharges') if c in df.columns]
        if num_cols:
            df[num_cols] = self.scaler.transform(df[num_cols])

        return df

    # ── single prediction ─────────────────────────────────────
    def predict_single(self, customer_dict):
        """
        Predict churn for one customer.
        Returns dict with probability, risk, and top contributing factors.
        """
        X = self._prepare_row(customer_dict)
        prob = float(self.model.predict_proba(X)[0][1])
        risk = self.risk_category(prob)
        factors = self._explain(X, prob)

        return {
            'probability': round(prob * 100, 1),
            'risk': risk,
            'factors': factors,
        }

    # ── explanation ────────────────────────────────────────────
    def _explain(self, X, prob):
        """Return a list of {'name', 'impact', 'direction'} dicts."""
        # Use coefficients (Logistic Regression) or feature importances
        if hasattr(self.model, 'coef_'):
            weights = self.model.coef_[0]
        elif hasattr(self.model, 'feature_importances_'):
            weights = self.model.feature_importances_
        else:
            return []

        contributions = []
        row = X.iloc[0].values
        for i, col in enumerate(self.feature_columns):
            val = row[i]
            w = weights[i]
            contrib = val * w
            if abs(contrib) > 0.01:
                label = self.FEATURE_LABELS.get(col, col.replace('_', ' '))
                contributions.append({
                    'name': label,
                    'impact': round(abs(contrib), 3),
                    'direction': 'negative' if contrib > 0 else 'positive',
                    'raw_col': col,
                })

        # Sort by absolute impact
        contributions.sort(key=lambda x: x['impact'], reverse=True)
        return contributions[:8]  # top 8 factors

    # ── bulk prediction ────────────────────────────────────────
    def predict_bulk(self, df):
        """
        Predict churn for an entire DataFrame.
        Returns the original df with added columns:
          Churn_Probability, Risk_Category
        """
        result = df.copy()

        # Save & drop customerID if present
        cid = None
        if 'customerID' in result.columns:
            cid = result['customerID'].copy()
            result.drop('customerID', axis=1, inplace=True)

        # Drop Churn if present
        if 'Churn' in result.columns:
            result.drop('Churn', axis=1, inplace=True)

        # Convert SeniorCitizen to category if needed
        if 'SeniorCitizen' in result.columns:
            result['SeniorCitizen'] = result['SeniorCitizen'].map(
                {0: 'No', 1: 'Yes', '0': 'No', '1': 'Yes'}
            ).fillna(result['SeniorCitizen'])

        # Numeric fixes
        for col in ('tenure', 'MonthlyCharges', 'TotalCharges'):
            if col in result.columns:
                result[col] = pd.to_numeric(
                    result[col].astype(str).str.strip(), errors='coerce'
                ).fillna(0)

        if 'TotalCharges' not in result.columns:
            result['TotalCharges'] = result.get('tenure', 0) * result.get('MonthlyCharges', 0)

        # One-hot encode
        num_cols_list = ['tenure', 'MonthlyCharges', 'TotalCharges']
        cat_cols = [c for c in result.columns if c not in num_cols_list]
        X = pd.get_dummies(result, columns=cat_cols, drop_first=True)

        # Align columns
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_columns]

        # Scale
        num_present = [c for c in num_cols_list if c in X.columns]
        if num_present:
            X[num_present] = self.scaler.transform(X[num_present])

        # Predict
        probs = self.model.predict_proba(X)[:, 1]

        # Build output
        out = df.copy()
        out['Churn_Probability'] = (probs * 100).round(1)
        out['Risk_Category'] = [self.risk_category(p) for p in probs]

        return out
