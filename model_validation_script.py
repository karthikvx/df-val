import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    classification_report, confusion_matrix, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import boto3
from datetime import datetime
import json
import os

class MortgageLoanModelValidator:
    
    def __init__(self, model_path, scaler_path, label_encoders_path):
        """Initialize the model validator"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoders = joblib.load(label_encoders_path)
        self.validation_results = {}
    
    def load_test_data(self, test_data_path):
        """Load and prepare test data"""
        self.test_df = pd.read_parquet(test_data_path)
        self.test_df = self._feature_engineering(self.test_df)
        self.X_test, self.y_test = self._prepare_features(self.test_df)
        
    def _feature_engineering(self, df):
        """Apply same feature engineering as training"""
        # Debt-to-income ratio
        df['debt_to_income_ratio'] = df['monthly_debt'] / df['monthly_income']
        
        # Loan-to-value ratio  
        df['loan_to_value_ratio'] = df['loan_amount'] / df['property_value']
        
        # Credit utilization
        df['credit_utilization'] = df['credit_used'] / df['credit_limit']
        
        # Payment-to-income ratio
        df['payment_to_income'] = df['monthly_payment'] / df['monthly_income']
        
        # Age calculations
        df['borrower_age'] = 2024 - df['birth_year']
        df['property_age'] = 2024 - df['year_built']
        
        # Combined credit score
        df['combined_credit_score'] = df[['primary_credit_score', 'secondary_credit_score']].mean(axis=1)
        
        return df
    
    def _prepare_features(self, df):
        """Prepare features using saved encoders"""
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
        
        # Apply saved label encoders
        for col in categorical_features:
            le = self.label_encoders[col]
            # Handle unseen categories
            df[col] = df[col].map(lambda x: x if x in le.classes_ else 'Unknown')
            df[col + '_encoded'] = le.transform(df[col])
        
        feature_columns = numerical_features + [col + '_encoded' for col in categorical_features]
        
        return df[feature_columns], df['default_flag']
    
    def statistical_validation(self):
        """Perform statistical validation tests"""
        print("=== STATISTICAL VALIDATION ===")
        
        # Scale features
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Get predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Basic metrics
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        avg_precision = average_precision_score(self.y_test, y_pred_proba)
        
        print(f"AUC-ROC Score: {auc_score:.4f}")
        print(f"Average Precision Score: {avg_precision:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        self.validation_results['statistical'] = {
            'auc_score': auc_score,
            'avg_precision': avg_precision,
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True),
            'confusion_matrix': cm.tolist()
        }
        
        return y_pred_proba
    
    def performance_stability_test(self, y_pred_proba):
        """Test model performance across different segments"""
        print("\n=== PERFORMANCE STABILITY TEST ===")
        
        # Test across different loan amounts
        loan_amount_bins = pd.qcut(self.test_df['loan_amount'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        
        stability_results = {}
        
        for segment in ['loan_amount_quartile', 'state', 'loan_purpose']:
            if segment == 'loan_amount_quartile':
                segments = loan_amount_bins
                segment_name = 'Loan Amount Quartile'
            else:
                segments = self.test_df[segment.replace('_quartile', '')]
                segment_name = segment.replace('_', ' ').title()
            
            print(f"\n{segment_name} Performance:")
            segment_results = {}
            
            for seg_value in segments.unique():
                if pd.isna(seg_value):
                    continue
                    
                mask = segments == seg_value
                if mask.sum() < 30:  # Skip segments with too few samples
                    continue
                
                seg_auc = roc_auc_score(self.y_test[mask], y_pred_proba[mask])
                seg_default_rate = self.y_test[mask].mean()
                
                print(f"  {seg_value}: AUC={seg_auc:.4f}, Default Rate={seg_default_rate:.4f}, N={mask.sum()}")
                
                segment_results[str(seg_value)] = {
                    'auc': seg_auc,
                    'default_rate': seg_default_rate,
                    'sample_size': int(mask.sum())
                }
            
            stability_results[segment] = segment_results
        
        self.validation_results['stability'] = stability_results
    
    def bias_fairness_test(self, y_pred_proba):
        """Test for potential bias in model predictions"""
        print("\n=== BIAS AND FAIRNESS TEST ===")
        
        bias_results = {}
        
        # Test across demographic segments (if available)
        if 'borrower_age' in self.test_df.columns:
            # Age groups
            age_groups = pd.cut(self.test_df['borrower_age'], 
                              bins=[0, 30, 45, 60, 100], 
                              labels=['Under 30', '30-45', '45-60', 'Over 60'])
            
            print("Age Group Analysis:")
            for age_group in age_groups.unique():
                if pd.isna(age_group):
                    continue
                    
                mask = age_groups == age_group
                if mask.sum() < 30:
                    continue
                
                # Average prediction probability
                avg_pred_prob = y_pred_proba[mask].mean()
                actual_default_rate = self.y_test[mask].mean()
                
                print(f"  {age_group}: Avg Predicted Risk={avg_pred_prob:.4f}, "
                      f"Actual Default Rate={actual_default_rate:.4f}")
                
                bias_results[f'age_group_{age_group}'] = {
                    'avg_predicted_prob': float(avg_pred_prob),
                    'actual_default_rate': float(actual_default_rate),
                    'sample_size': int(mask.sum())
                }
        
        self.validation_results['bias'] = bias_results
    
    def calibration_test(self, y_pred_proba):
        """Test model calibration"""
        print("\n=== CALIBRATION TEST ===")
        
        # Bin predictions and check calibration
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_results = []
        
        print("Calibration Analysis:")
        print("Predicted Risk Range | Actual Default Rate | Count")
        print("-" * 50)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = self.y_test[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                count_in_bin = in_bin.sum()
                
                print(f"{bin_lower:.2f} - {bin_upper:.2f}        | {accuracy_in_bin:.4f}           | {count_in_bin}")
                
                calibration_results.append({
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'avg_predicted': float(avg_confidence_in_bin),
                    'actual_rate': float(accuracy_in_bin),
                    'count': int(count_in_bin)
                })
        
        self.validation_results['calibration'] = calibration_results
    
    def business_validation(self, y_pred_proba):
        """Business-specific validation tests"""
        print("\n=== BUSINESS VALIDATION ===")
        
        # Test different risk thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        print("Risk Threshold Analysis:")
        print("Threshold | Precision | Recall | False Positive Rate | Loans Approved")
        print("-" * 70)
        
        threshold_results = []
        
        for threshold in thresholds:
            y_pred_threshold = (y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics
            precision = ((y_pred_threshold == 1) & (self.y_test == 1)).sum() / (y_pred_threshold == 1).sum()
            recall = ((y_pred_threshold == 1) & (self.y_test == 1)).sum() / (self.y_test == 1).sum()
            fpr = ((y_pred_threshold == 1) & (self.y_test == 0)).sum() / (self.y_test == 0).sum()
            approval_rate = (y_pred_threshold == 0).mean()  # Loans approved (predicted as not default)
            
            print(f"{threshold:.1f}       | {precision:.4f}    | {recall:.4f} | {fpr:.4f}             | {approval_rate:.4f}")
            
            threshold_results.append({
                'threshold': threshold,
                'precision': float(precision),
                'recall': float(recall),
                'false_positive_rate': float(fpr),
                'approval_rate': float(approval_rate)
            })
        
        self.validation_results['business'] = threshold_results
    
    def generate_validation_report(self, output_path):
        """Generate comprehensive validation report"""
        
        # Add metadata
        self.validation_results['metadata'] = {
            'validation_date': datetime.now().isoformat(),
            'test_sample_size': int(len(self.y_test)),
            'test_default_rate': float(self.y_test.mean()),
            'model_type': str(type(self.model).__name__)
        }
        
        # Save results to JSON
        with open(os.path.join(output_path, 'validation_results.json'), 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        # Create summary report
        summary_report = f"""
MORTGAGE LOAN DEFAULT MODEL VALIDATION REPORT
=============================================

Validation Date: {self.validation_results['metadata']['validation_date']}
Model Type: {self.validation_results['metadata']['model_type']}

Test Dataset:
- Sample Size: {self.validation_results['metadata']['test_sample_size']:,}
- Default Rate: {self.validation_results['metadata']['test_default_rate']:.4f}

Statistical Performance:
- AUC-ROC Score: {self.validation_results['statistical']['auc_score']:.4f}
- Average Precision: {self.validation_results['statistical']['avg_precision']:.4f}

Recommended Business Threshold: 0.3
- Expected Precision: {[r for r in self.validation_results['business'] if r['threshold'] == 0.3][0]['precision']:.4f}
- Expected Recall: {[r for r in self.validation_results['business'] if r['threshold'] == 0.3][0]['recall']:.4f}
- Expected Approval Rate: {[r for r in self.validation_results['business'] if r['threshold'] == 0.3][0]['approval_rate']:.4f}

Model Status: {'PASSED' if self.validation_results['statistical']['auc_score'] > 0.7 else 'NEEDS REVIEW'}
        """
        
        with open(os.path.join(output_path, 'validation_summary.txt'), 'w') as f:
            f.write(summary_report)
        
        print(summary_report)
        
        return self.validation_results
    
    def run_full_validation(self, test_data_path, output_path):
        """Run complete validation pipeline"""
        
        print("Starting Model Validation Pipeline...")
        
        # Load test data
        self.load_test_data(test_data_path)
        
        # Run all validation tests
        y_pred_proba = self.statistical_validation()
        self.performance_stability_test(y_pred_proba)
        self.bias_fairness_test(y_pred_proba)
        self.calibration_test(y_pred_proba)
        self.business_validation(y_pred_proba)
        
        # Generate report
        results = self.generate_validation_report(output_path)
        
        print(f"\nValidation complete! Results saved to {output_path}")
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize validator
    validator = MortgageLoanModelValidator(
        model_path='/opt/ml/model/model.joblib',
        scaler_path='/opt/ml/model/scaler.joblib', 
        label_encoders_path='/opt/ml/model/label_encoders.joblib'
    )
    
    # Run validation
    results = validator.run_full_validation(
        test_data_path='/opt/ml/input/data/test',
        output_path='/opt/ml/output'
    )