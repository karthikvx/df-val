import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import boto3
import os
import argparse

def load_data(data_path):
    """Load and prepare mortgage loan data"""
    df = pd.read_parquet(data_path)
    return df

def feature_engineering(df):
    """Create relevant features for mortgage default prediction"""
    
    # Debt-to-income ratio
    df['debt_to_income_ratio'] = df['monthly_debt'] / df['monthly_income'].replace(0, np.nan)
    
    # Loan-to-value ratio
    df['loan_to_value_ratio'] = df['loan_amount'] / df['property_value'].replace(0, np.nan)
    
    # Credit utilization
    df['credit_utilization'] = df['credit_used'] / df['credit_limit'].replace(0, np.nan)
    
    # Payment-to-income ratio
    df['payment_to_income'] = df['monthly_payment'] / df['monthly_income'].replace(0, np.nan)
    
    # Age of borrower
    df['borrower_age'] = 2024 - df['birth_year']
    
    # Property age
    df['property_age'] = 2024 - df['year_built']
    
    # Combined credit score (if multiple borrowers)
    df['combined_credit_score'] = df[['primary_credit_score', 'secondary_credit_score']].mean(axis=1)
    
    return df

def prepare_features(df):
    """Prepare features for modeling"""
    
    # Define feature columns
    numerical_features = [
        'loan_amount', 'property_value', 'monthly_income', 'monthly_debt',
        'credit_score', 'loan_term', 'interest_rate', 'down_payment',
        'debt_to_income_ratio', 'loan_to_value_ratio', 'credit_utilization',
        'payment_to_income', 'borrower_age', 'property_age', 'combined_credit_score'
    ]
    
    categorical_features = [
        'loan_purpose', 'property_type', 'occupancy_type', 
        'employment_type', 'state', 'loan_type'
    ]
    
    # Handle missing values
    df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())
    df[categorical_features] = df[categorical_features].fillna('Unknown')
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Feature selection
    feature_columns = numerical_features + [col + '_encoded' for col in categorical_features]
    
    return df[feature_columns], df['default_flag'], label_encoders

def train_model(X_train, y_train, X_val, y_val):
    """Train mortgage default risk models"""
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    models = {}
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train_scaled, y_train)
    models['random_forest'] = rf_model
    
    # Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    gb_model.fit(X_train_scaled, y_train)
    models['gradient_boosting'] = gb_model
    
    return models, scaler

def evaluate_models(models, scaler, X_val, y_val):
    """Evaluate model performance"""
    
    X_val_scaled = scaler.transform(X_val)
    results = {}
    
    for name, model in models.items():
        # Predictions
        y_pred = model.predict(X_val_scaled)
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        
        # Metrics
        auc_score = roc_auc_score(y_val, y_pred_proba)
        
        results[name] = {
            'auc_score': auc_score,
            'classification_report': classification_report(y_val, y_pred),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"\n{name.upper()} Results:")
        print(f"AUC Score: {auc_score:.4f}")
        print(results[name]['classification_report'])
    
    return results

def save_model_artifacts(best_model, scaler, label_encoders, feature_columns, model_dir):
    """Save model and preprocessing artifacts"""
    
    # Save model
    joblib.dump(best_model, os.path.join(model_dir, 'model.joblib'))
    
    # Save scaler
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    
    # Save label encoders
    joblib.dump(label_encoders, os.path.join(model_dir, 'label_encoders.joblib'))
    
    # Save feature columns
    joblib.dump(feature_columns, os.path.join(model_dir, 'feature_columns.joblib'))
    
    print(f"Model artifacts saved to {model_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='/opt/ml/input/data/training')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')
    parser.add_argument('--output-dir', type=str, default='/opt/ml/output')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    df = load_data(args.data_path)
    
    # Feature engineering
    print("Engineering features...")
    df = feature_engineering(df)
    
    # Prepare features
    print("Preparing features...")
    X, y, label_encoders = prepare_features(df)
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Default rate in training: {y_train.mean():.4f}")
    
    # Train models
    print("Training models...")
    models, scaler = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate models
    print("Evaluating models...")
    results = evaluate_models(models, scaler, X_val, y_val)
    
    # Select best model (highest AUC)
    best_model_name = max(results.keys(), key=lambda k: results[k]['auc_score'])
    best_model = models[best_model_name]
    
    print(f"\nBest model: {best_model_name} (AUC: {results[best_model_name]['auc_score']:.4f})")
    
    # Save model artifacts
    save_model_artifacts(
        best_model, scaler, label_encoders, 
        X.columns.tolist(), args.model_dir
    )
    
    # Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        # Save feature importance
        feature_importance.to_csv(
            os.path.join(args.output_dir, 'feature_importance.csv'), 
            index=False
        )

if __name__ == '__main__':
    main()