import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.model_monitor import DefaultModelMonitor
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MortgageRiskModelTraining:
    def __init__(self, role_arn, bucket_name, region='us-east-1'):
        """
        Initialize SageMaker training configuration for mortgage risk model
        
        Args:
            role_arn: AWS IAM role ARN with SageMaker permissions
            bucket_name: S3 bucket for storing artifacts
            region: AWS region
        """
        self.role_arn = role_arn
        self.bucket_name = bucket_name
        self.region = region
        
        # Initialize SageMaker session
        self.sagemaker_session = sagemaker.Session(default_bucket=bucket_name)
        self.boto_session = boto3.Session(region_name=region)
        
        # Set up S3 paths
        self.s3_prefix = 'mortgage-risk-model'
        self.input_path = f's3://{bucket_name}/{self.s3_prefix}/input'
        self.output_path = f's3://{bucket_name}/{self.s3_prefix}/output'
        self.code_path = f's3://{bucket_name}/{self.s3_prefix}/code'
        self.model_path = f's3://{bucket_name}/{self.s3_prefix}/models'
        
        # Training configuration
        self.training_config = {
            'instance_type': 'ml.m5.2xlarge',  # 8 vCPU, 32 GB RAM
            'instance_count': 1,
            'volume_size': 30,  # GB
            'max_run': 3600,  # 1 hour timeout
            'framework_version': '1.2-1',
            'python_version': 'py3'
        }
        
        logger.info(f"Initialized MortgageRiskModelTraining for bucket: {bucket_name}")

    def prepare_training_data(self, raw_data_path, validation_split=0.2):
        """
        Prepare and preprocess training data using SageMaker Processing
        
        Args:
            raw_data_path: S3 path to raw mortgage data
            validation_split: Fraction of data for validation
        """
        logger.info("Starting data preparation...")
        
        # Create preprocessing script
        preprocessing_script = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import argparse
import os

def preprocess_mortgage_data(input_path, output_path, validation_split):
    # Load data
    df = pd.read_csv(input_path)
    
    # Feature engineering (simplified version)
    # Calculate derived features
    df['debt_to_income'] = df['total_debt'] / df['income']
    df['loan_to_value'] = df['loan_amount'] / df['property_value']
    df['credit_utilization'] = df['credit_card_balance'] / df['credit_limit']
    df['payment_to_income'] = df['monthly_payment'] / (df['income'] / 12)
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        if col != 'default':  # Assuming 'default' is target
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Split features and target
    target_col = 'default'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save processed data
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(
        os.path.join(output_path, 'train_features.csv'), index=False
    )
    pd.DataFrame({'target': y_train}).to_csv(
        os.path.join(output_path, 'train_target.csv'), index=False
    )
    pd.DataFrame(X_val_scaled, columns=X_val.columns).to_csv(
        os.path.join(output_path, 'val_features.csv'), index=False
    )
    pd.DataFrame({'target': y_val}).to_csv(
        os.path.join(output_path, 'val_target.csv'), index=False
    )
    
    # Save preprocessing artifacts
    joblib.dump(scaler, os.path.join(output_path, 'scaler.pkl'))
    joblib.dump(label_encoders, os.path.join(output_path, 'label_encoders.pkl'))
    
    # Save feature names
    with open(os.path.join(output_path, 'feature_names.json'), 'w') as f:
        json.dump(list(X_train.columns), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, required=True)
    parser.add_argument('--output-data', type=str, required=True)
    parser.add_argument('--validation-split', type=float, default=0.2)
    
    args = parser.parse_args()
    
    preprocess_mortgage_data(
        args.input_data,
        args.output_data,
        args.validation_split
    )
"""
        
        # Save preprocessing script to S3
        script_path = 'preprocessing.py'
        with open(script_path, 'w') as f:
            f.write(preprocessing_script)
        
        # Upload to S3
        s3_script_path = f"{self.code_path}/preprocessing.py"
        self.sagemaker_session.upload_data(script_path, bucket=self.bucket_name, 
                                          key_prefix=f"{self.s3_prefix}/code")
        
        # Create SKLearn processor
        processor = SKLearnProcessor(
            framework_version='1.2-1',
            role=self.role_arn,
            instance_type='ml.m5.large',
            instance_count=1,
            base_job_name='mortgage-data-preprocessing',
            sagemaker_session=self.sagemaker_session
        )
        
        # Run processing job
        processor.run(
            code='preprocessing.py',
            source_dir=None,
            inputs=[ProcessingInput(
                source=raw_data_path,
                destination='/opt/ml/processing/input'
            )],
            outputs=[ProcessingOutput(
                source='/opt/ml/processing/output',
                destination=f"{self.input_path}/processed"
            )],
            arguments=[
                '--input-data', '/opt/ml/processing/input',
                '--output-data', '/opt/ml/processing/output',
                '--validation-split', str(validation_split)
            ]
        )
        
        logger.info("Data preprocessing completed")
        return f"{self.input_path}/processed"

    def create_training_job(self, processed_data_path):
        """
        Create and launch SageMaker training job
        
        Args:
            processed_data_path: S3 path to processed training data
        """
        logger.info("Creating SageMaker training job...")
        
        # Training hyperparameters
        hyperparameters = {
            'random_forest_params': json.dumps({
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }),
            'gradient_boost_params': json.dumps({
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            }),
            'cross_validation_folds': 5,
            'test_size': 0.2,
            'enable_feature_importance': True,
            'model_selection_metric': 'roc_auc'
        }
        
        # Create SKLearn estimator
        sklearn_estimator = SKLearn(
            entry_point='train.py',  # Our training script
            source_dir='scripts',    # Local directory with training code
            role=self.role_arn,
            instance_type=self.training_config['instance_type'],
            instance_count=self.training_config['instance_count'],
            framework_version=self.training_config['framework_version'],
            py_version=self.training_config['python_version'],
            hyperparameters=hyperparameters,
            output_path=self.output_path,
            base_job_name='mortgage-risk-training',
            max_run=self.training_config['max_run'],
            volume_size=self.training_config['volume_size'],
            sagemaker_session=self.sagemaker_session,
            environment={
                'AWS_DEFAULT_REGION': self.region,
                'SAGEMAKER_PROGRAM': 'train.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
            }
        )
        
        # Training data inputs
        training_inputs = {
            'training': processed_data_path,
            'validation': processed_data_path  # Same path, different files
        }
        
        # Start training
        sklearn_estimator.fit(training_inputs, wait=False)
        
        logger.info(f"Training job started: {sklearn_estimator.latest_training_job.name}")
        return sklearn_estimator

    def create_model_package(self, estimator, model_approval_status='PendingManualApproval'):
        """
        Create a model package for deployment
        
        Args:
            estimator: Trained SageMaker estimator
            model_approval_status: Approval status for the model package
        """
        logger.info("Creating model package...")
        
        model_package_group_name = 'mortgage-risk-model-group'
        
        # Create model package
        model_package = estimator.register(
            content_types=['text/csv'],
            response_types=['text/csv'],
            inference_instances=['ml.t2.medium', 'ml.m5.large'],
            transform_instances=['ml.m5.large'],
            model_package_group_name=model_package_group_name,
            approval_status=model_approval_status,
            description='Mortgage loan default risk prediction model',
            model_metrics={
                'classification_metrics': {
                    'auc': {
                        'value': 0.85,  # Will be updated with actual metrics
                        'standard_deviation': 0.02
                    },
                    'precision': {
                        'value': 0.78
                    },
                    'recall': {
                        'value': 0.82
                    }
                }
            }
        )
        
        logger.info(f"Model package created: {model_package.model_package_arn}")
        return model_package

    def setup_model_monitoring(self, endpoint_name):
        """
        Set up model monitoring for deployed model
        
        Args:
            endpoint_name: Name of the deployed endpoint
        """
        logger.info("Setting up model monitoring...")
        
        # Create model monitor
        model_monitor = DefaultModelMonitor(
            role=self.role_arn,
            instance_count=1,
            instance_type='ml.m5.large',
            volume_size_in_gb=20,
            max_runtime_in_seconds=3600,
            sagemaker_session=self.sagemaker_session
        )
        
        # Baseline configuration
        baseline_results = model_monitor.suggest_baseline(
            baseline_dataset=f"{self.input_path}/processed/baseline.csv",
            dataset_format='csv',
            output_s3_uri=f"{self.output_path}/monitoring/baseline",
            wait=True
        )
        
        # Create monitoring schedule
        model_monitor.create_monitoring_schedule(
            monitor_schedule_name=f"{endpoint_name}-monitoring",
            endpoint_input=endpoint_name,
            output_s3_uri=f"{self.output_path}/monitoring/results",
            statistics=baseline_results.baseline_statistics,
            constraints=baseline_results.baseline_constraints,
            schedule_cron_expression='cron(0 8 * * ? *)',  # Daily at 8 AM
            enable_cloudwatch_metrics=True
        )
        
        logger.info("Model monitoring setup completed")
        return model_monitor

    def run_full_pipeline(self, raw_data_path):
        """
        Run the complete training pipeline
        
        Args:
            raw_data_path: S3 path to raw mortgage data
        """
        try:
            logger.info("Starting full ML pipeline...")
            
            # Step 1: Data preparation
            processed_data_path = self.prepare_training_data(raw_data_path)
            
            # Step 2: Model training
            estimator = self.create_training_job(processed_data_path)
            
            # Wait for training to complete
            estimator.latest_training_job.wait()
            
            # Step 3: Create model package
            model_package = self.create_model_package(estimator)
            
            # Step 4: Deploy for testing (optional)
            # predictor = estimator.deploy(
            #     initial_instance_count=1,
            #     instance_type='ml.t2.medium',
            #     endpoint_name='mortgage-risk-model-test'
            # )
            
            logger.info("ML pipeline completed successfully")
            
            return {
                'training_job_name': estimator.latest_training_job.name,
                'model_package_arn': model_package.model_package_arn,
                'model_artifacts': estimator.model_data
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

def main():
    """
    Main execution function - example usage
    """
    # Configuration
    ROLE_ARN = 'arn:aws:iam::123456789012:role/SageMakerExecutionRole'
    BUCKET_NAME = 'mortgage-risk-ml-bucket'
    RAW_DATA_PATH = 's3://mortgage-risk-ml-bucket/raw-data/mortgage_training_data.parquet'
    TEST_DATA_PATH = 's3://mortgage-risk-ml-bucket/raw-data/mortgage_test_data.parquet'
    
    # Initialize training pipeline
    training_pipeline = MortgageRiskModelTraining(
        role_arn=ROLE_ARN,
        bucket_name=BUCKET_NAME
    )
    
    # Run complete pipeline with validation
    results = training_pipeline.run_full_pipeline(
        raw_data_path=RAW_DATA_PATH,
        test_data_path=TEST_DATA_PATH
    )
    
    print("Training Pipeline Results:")
    print(json.dumps(results, indent=2))
    
    # Optional: Deploy model for real-time inference
    # if input("Deploy model for real-time inference? (y/n): ").lower() == 'y':
    #     deploy_model_for_inference(results['model_package_arn'])

def deploy_model_for_inference(model_package_arn):
    """
    Deploy model for real-time inference
    
    Args:
        model_package_arn: ARN of the model package to deploy
    """
    sagemaker_session = sagemaker.Session()
    
    # Create model from package
    model = sagemaker.model.ModelPackage(
        role='arn:aws:iam::123456789012:role/SageMakerExecutionRole',
        model_package_arn=model_package_arn,
        sagemaker_session=sagemaker_session
    )
    
    # Deploy model
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.t2.medium',
        endpoint_name='mortgage-risk-model-prod'
    )
    
    print(f"Model deployed to endpoint: {predictor.endpoint_name}")
    return predictor

if __name__ == "__main__":
    main()
