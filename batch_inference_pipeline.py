import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.transformer import Transformer
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MortgageBatchInferencePipeline:
    
    def __init__(self, role_arn, bucket_name, model_package_arn=None, 
                 model_data_path=None, region='us-east-1'):
        """
        Initialize batch inference pipeline for mortgage risk model
        
        Args:
            role_arn: AWS IAM role ARN with SageMaker permissions
            bucket_name: S3 bucket for storing batch inference data
            model_package_arn: ARN of registered model package (optional)
            model_data_path: S3 path to model artifacts (optional)
            region: AWS region
        """
        self.role_arn = role_arn
        self.bucket_name = bucket_name
        self.model_package_arn = model_package_arn
        self.model_data_path = model_data_path
        self.region = region
        
        # Initialize AWS sessions
        self.sagemaker_session = sagemaker.Session(default_bucket=bucket_name)
        self.boto_session = boto3.Session(region_name=region)
        self.s3_client = self.boto_session.client('s3')
        
        # Set up S3 paths
        self.s3_prefix = 'mortgage-batch-inference'
        self.input_path = f's3://{bucket_name}/{self.s3_prefix}/input'
        self.output_path = f's3://{bucket_name}/{self.s3_prefix}/output'
        self.temp_path = f's3://{bucket_name}/{self.s3_prefix}/temp'
        
        logger.info("Initialized MortgageBatchInferencePipeline")

    def create_batch_transform_script(self):
        """Create batch transform script for preprocessing and inference"""
        
        transform_script = '''
import pandas as pd
import numpy as np
import joblib
import json
import os
import argparse
from datetime import datetime

def feature_engineering(df):
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
    current_year = datetime.now().year
    df['borrower_age'] = current_year - df['birth_year']
    df['property_age'] = current_year - df['year_built']
    
    # Combined credit score
    df['combined_credit_score'] = df[['primary_credit_score', 'secondary_credit_score']].mean(axis=1)
    
    return df

def prepare_features_for_inference(df, label_encoders, expected_columns):
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
        if col in label_encoders:
            le = label_encoders[col]
            # Handle unseen categories
            df[col] = df[col].map(lambda x: x if x in le.classes_ else 'Unknown')
            
            # Handle case where 'Unknown' is not in training classes
            if 'Unknown' not in le.classes_:
                most_frequent = le.classes_[0]
                df[col] = df[col].map(lambda x: x if x in le.classes_ else most_frequent)
            
            df[col + '_encoded'] = le.transform(df[col])
    
    # Select expected features in correct order
    feature_columns = numerical_features + [col + '_encoded' for col in categorical_features]
    
    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0  # Default value for missing columns
    
    return df[expected_columns]

def get_risk_category(probability):
    """Convert probability to risk category"""
    if probability < 0.2:
        return "LOW"
    elif probability < 0.4:
        return "MEDIUM"
    elif probability < 0.6:
        return "HIGH"
    else:
        return "VERY_HIGH"

def batch_predict(input_file, output_file, model_dir):
    """Perform batch prediction"""
    
    # Load model artifacts
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    label_encoders = joblib.load(os.path.join(model_dir, "label_encoders.joblib"))
    feature_columns = joblib.load(os.path.join(model_dir, "feature_columns.joblib"))
    
    print(f"Loading data from {input_file}")
    
    # Load input data
    if input_file.endswith('.parquet'):
        df = pd.read_parquet(input_file)
    elif input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        raise ValueError("Unsupported file format. Use CSV or Parquet.")
    
    print(f"Loaded {len(df)} records for prediction")
    
    # Store original loan IDs
    loan_ids = df.get('loan_id', [f'loan_{i}' for i in range(len(df))]).tolist()
    
    # Feature engineering
    df_processed = feature_engineering(df.copy())
    
    # Prepare features
    X = prepare_features_for_inference(df_processed, label_encoders, feature_columns)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Prepare results
    results = []
    for i in range(len(X)):
        result = {
            'loan_id': str(loan_ids[i]),
            'default_probability': float(probabilities[i][1]),
            'risk_prediction': int(predictions[i]),
            'risk_category': get_risk_category(probabilities[i][1]),
            'confidence_score': float(max(probabilities[i])),
            'prediction_timestamp': datetime.now().isoformat()
        }
        
        # Add original loan data for reference
        original_data = df.iloc[i].to_dict()
        result['original_data'] = {k: (v if pd.notna(v) else None) for k, v in original_data.items()}
        
        results.append(result)
    
    # Save results
    results_df = pd.DataFrame([
        {
            'loan_id': r['loan_id'],
            'default_probability': r['default_probability'],
            'risk_prediction': r['risk_prediction'],
            'risk_category': r['risk_category'],
            'confidence_score': r['confidence_score'],
            'prediction_timestamp': r['prediction_timestamp']
        }
        for r in results
    ])
    
    # Save detailed results as JSON
    with open(output_file.replace('.csv', '_detailed.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save summary CSV
    results_df.to_csv(output_file, index=False)
    
    print(f"Predictions saved to {output_file}")
    print(f"Detailed results saved to {output_file.replace('.csv', '_detailed.json')}")
    
    # Generate summary statistics
    summary = {
        'total_loans': len(results_df),
        'high_risk_loans': len(results_df[results_df['risk_category'].isin(['HIGH', 'VERY_HIGH'])]),
        'average_default_probability': float(results_df['default_probability'].mean()),
        'risk_distribution': results_df['risk_category'].value_counts().to_dict(),
        'processing_timestamp': datetime.now().isoformat()
    }
    
    with open(output_file.replace('.csv', '_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--model-dir', type=str, required=True)
    
    args = parser.parse_args()
    
    batch_predict(args.input_file, args.output_file, args.model_dir)
'''
        
        # Save transform script
        with open('batch_transform.py', 'w') as f:
            f.write(transform_script)
        
        logger.info("Batch transform script created")
        return 'batch_transform.py'

    def run_batch_inference_processing(self, input_data_path, output_prefix=None):
        """Run batch inference using SageMaker Processing"""
        
        if output_prefix is None:
            output_prefix = f"batch-{int(datetime.now().timestamp())}"
        
        # Create processing script
        script_path = self.create_batch_transform_script()
        
        # Create SKLearn processor
        # Create SKLearn processor
        processor = SKLearnProcessor(
            framework_version='1.2-1',
            role=self.role_arn,
            instance_type='ml.m5.2xlarge',
            instance_count=1,
            base_job_name='mortgage-batch-inference',
            sagemaker_session=self.sagemaker_session
        )
        
        try:
            # Run processing job
            processor.run(
                code=script_path,
                inputs=[
                    ProcessingInput(
                        source=input_data_path,
                        destination='/opt/ml/processing/input'
                    ),
                    ProcessingInput(
                        source=self.model_data_path,
                        destination='/opt/ml/processing/model'
                    )
                ],
                outputs=[
                    ProcessingOutput(
                        source='/opt/ml/processing/output',
                        destination=f'{self.output_path}/{output_prefix}'
                    )
                ],
                arguments=[
                    '--input-file', '/opt/ml/processing/input/input.csv',
                    '--output-file', '/opt/ml/processing/output/predictions.csv',
                    '--model-dir', '/opt/ml/processing/model'
                ]
            )
            
            logger.info(f"Batch inference job completed. Results at: {self.output_path}/{output_prefix}")
            return f'{self.output_path}/{output_prefix}'
            
        except Exception as e:
            logger.error(f"Batch inference processing failed: {str(e)}")
            raise

    def run_batch_transform_job(self, input_data_path, output_prefix=None):
        """Run batch inference using SageMaker Batch Transform"""
        
        if output_prefix is None:
            output_prefix = f"batch-transform-{int(datetime.now().timestamp())}"
        
        try:
            # Create model from package or artifacts
            if self.model_package_arn:
                model = sagemaker.ModelPackage(
                    role=self.role_arn,
                    model_package_arn=self.model_package_arn,
                    sagemaker_session=self.sagemaker_session
                )
            else:
                model = SKLearnModel(
                    model_data=self.model_data_path,
                    role=self.role_arn,
                    entry_point='batch_transform.py',
                    framework_version='1.2-1',
                    sagemaker_session=self.sagemaker_session
                )
            
            # Create transformer
            transformer = Transformer(
                model_name=model.name,
                instance_count=1,
                instance_type='ml.m5.2xlarge',
                output_path=f'{self.output_path}/{output_prefix}',
                sagemaker_session=self.sagemaker_session,
                accept='text/csv',
                assemble_with='Line'
            )
            
            # Run batch transform
            transformer.transform(
                data=input_data_path,
                content_type='text/csv',
                split_type='Line'
            )
            
            logger.info(f"Batch transform job completed. Results at: {self.output_path}/{output_prefix}")
            return f'{self.output_path}/{output_prefix}'
            
        except Exception as e:
            logger.error(f"Batch transform job failed: {str(e)}")
            raise

    def create_scheduled_inference_job(self, input_data_path, schedule_expression, 
                                     job_name=None, notification_topic=None):
        """Create scheduled batch inference job using EventBridge and Lambda"""
        
        if job_name is None:
            job_name = f"mortgage-batch-inference-{int(datetime.now().timestamp())}"
        
        # Create Lambda function for batch inference
        lambda_code = f'''
import json
import boto3
import logging
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """Lambda function to trigger batch inference"""
    
    try:
        # Initialize batch inference pipeline
        pipeline = MortgageBatchInferencePipeline(
            role_arn="{self.role_arn}",
            bucket_name="{self.bucket_name}",
            model_data_path="{self.model_data_path}",
            region="{self.region}"
        )
        
        # Get input data path from event or use default
        input_path = event.get('input_path', '{input_data_path}')
        output_prefix = event.get('output_prefix', f'scheduled-{{int(datetime.now().timestamp())}}')
        
        # Run batch inference
        result_path = pipeline.run_batch_inference_processing(input_path, output_prefix)
        
        # Send notification if topic provided
        if "{notification_topic}":
            sns = boto3.client('sns')
            sns.publish(
                TopicArn="{notification_topic}",
                Subject="Mortgage Batch Inference Completed",
                Message=f"Batch inference job completed successfully. Results at: {{result_path}}"
            )
        
        return {{
            'statusCode': 200,
            'body': json.dumps({{
                'message': 'Batch inference completed successfully',
                'result_path': result_path,
                'timestamp': datetime.now().isoformat()
            }})
        }}
        
    except Exception as e:
        logger.error(f"Batch inference failed: {{str(e)}}")
        
        if "{notification_topic}":
            sns = boto3.client('sns')
            sns.publish(
                TopicArn="{notification_topic}",
                Subject="Mortgage Batch Inference Failed",
                Message=f"Batch inference job failed: {{str(e)}}"
            )
        
        return {{
            'statusCode': 500,
            'body': json.dumps({{
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }})
        }}
'''
        
        # Save Lambda code
        with open(f'{job_name}_lambda.py', 'w') as f:
            f.write(lambda_code)
        
        logger.info(f"Scheduled inference job configuration created: {job_name}")
        return f'{job_name}_lambda.py'

    def validate_batch_results(self, results_path, validation_config=None):
        """Validate batch inference results"""
        
        if validation_config is None:
            validation_config = {
                'max_default_probability': 0.95,
                'min_confidence_score': 0.6,
                'expected_risk_distribution': {
                    'LOW': 0.4, 'MEDIUM': 0.3, 'HIGH': 0.2, 'VERY_HIGH': 0.1
                }
            }
        
        try:
            # Read results
            results_df = pd.read_csv(f'{results_path}/predictions.csv')
            
            validation_results = {
                'total_predictions': len(results_df),
                'validation_timestamp': datetime.now().isoformat(),
                'checks_passed': [],
                'checks_failed': [],
                'warnings': []
            }
            
            # Check 1: Probability range validation
            invalid_probs = results_df[
                (results_df['default_probability'] < 0) | 
                (results_df['default_probability'] > 1)
            ]
            
            if len(invalid_probs) == 0:
                validation_results['checks_passed'].append('probability_range')
            else:
                validation_results['checks_failed'].append({
                    'check': 'probability_range',
                    'issue': f'{len(invalid_probs)} predictions with invalid probabilities'
                })
            
            # Check 2: Extreme probability values
            extreme_probs = results_df[
                results_df['default_probability'] > validation_config['max_default_probability']
            ]
            
            if len(extreme_probs) == 0:
                validation_results['checks_passed'].append('extreme_probabilities')
            else:
                validation_results['warnings'].append({
                    'check': 'extreme_probabilities',
                    'issue': f'{len(extreme_probs)} predictions with very high default probability'
                })
            
            # Check 3: Confidence score validation
            low_confidence = results_df[
                results_df['confidence_score'] < validation_config['min_confidence_score']
            ]
            
            if len(low_confidence) == 0:
                validation_results['checks_passed'].append('confidence_scores')
            else:
                validation_results['warnings'].append({
                    'check': 'confidence_scores',
                    'issue': f'{len(low_confidence)} predictions with low confidence'
                })
            
            # Check 4: Risk distribution validation
            actual_distribution = results_df['risk_category'].value_counts(normalize=True).to_dict()
            expected_dist = validation_config['expected_risk_distribution']
            
            distribution_check = all(
                abs(actual_distribution.get(category, 0) - expected_pct) < 0.2
                for category, expected_pct in expected_dist.items()
            )
            
            if distribution_check:
                validation_results['checks_passed'].append('risk_distribution')
            else:
                validation_results['warnings'].append({
                    'check': 'risk_distribution',
                    'issue': 'Risk distribution differs significantly from expected',
                    'actual': actual_distribution,
                    'expected': expected_dist
                })
            
            # Check 5: Missing predictions
            missing_predictions = results_df[results_df['default_probability'].isna()]
            
            if len(missing_predictions) == 0:
                validation_results['checks_passed'].append('missing_predictions')
            else:
                validation_results['checks_failed'].append({
                    'check': 'missing_predictions',
                    'issue': f'{len(missing_predictions)} missing predictions'
                })
            
            # Overall validation status
            validation_results['overall_status'] = (
                'PASSED' if len(validation_results['checks_failed']) == 0 else 'FAILED'
            )
            
            # Save validation results
            with open(f'{results_path}/validation_results.json', 'w') as f:
                json.dump(validation_results, f, indent=2)
            
            logger.info(f"Batch results validation completed: {validation_results['overall_status']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Batch results validation failed: {str(e)}")
            raise

    def generate_batch_report(self, results_path, report_config=None):
        """Generate comprehensive batch inference report"""
        
        if report_config is None:
            report_config = {
                'include_charts': True,
                'include_risk_analysis': True,
                'include_performance_metrics': True
            }
        
        try:
            # Read results and summary
            results_df = pd.read_csv(f'{results_path}/predictions.csv')
            
            with open(f'{results_path}/predictions_summary.json', 'r') as f:
                summary = json.load(f)
            
            # Generate report
            report = {
                'report_metadata': {
                    'generation_timestamp': datetime.now().isoformat(),
                    'total_predictions': len(results_df),
                    'report_version': '1.0'
                },
                
                'executive_summary': {
                    'total_loans_processed': summary['total_loans'],
                    'high_risk_loans': summary['high_risk_loans'],
                    'high_risk_percentage': round(summary['high_risk_loans'] / summary['total_loans'] * 100, 2),
                    'average_default_probability': summary['average_default_probability'],
                    'risk_distribution': summary['risk_distribution']
                },
                
                'detailed_analytics': {
                    'probability_statistics': {
                        'mean': float(results_df['default_probability'].mean()),
                        'median': float(results_df['default_probability'].median()),
                        'std': float(results_df['default_probability'].std()),
                        'min': float(results_df['default_probability'].min()),
                        'max': float(results_df['default_probability'].max()),
                        'percentiles': {
                            '25th': float(results_df['default_probability'].quantile(0.25)),
                            '75th': float(results_df['default_probability'].quantile(0.75)),
                            '90th': float(results_df['default_probability'].quantile(0.90)),
                            '95th': float(results_df['default_probability'].quantile(0.95))
                        }
                    },
                    
                    'confidence_analysis': {
                        'mean_confidence': float(results_df['confidence_score'].mean()),
                        'low_confidence_count': len(results_df[results_df['confidence_score'] < 0.7]),
                        'high_confidence_count': len(results_df[results_df['confidence_score'] > 0.9])
                    },
                    
                    'risk_category_breakdown': {
                        category: {
                            'count': int(count),
                            'percentage': round(count / len(results_df) * 100, 2),
                            'avg_probability': float(results_df[results_df['risk_category'] == category]['default_probability'].mean())
                        }
                        for category, count in results_df['risk_category'].value_counts().items()
                    }
                }
            }
            
            # Add business impact analysis if requested
            if report_config.get('include_risk_analysis', False):
                report['business_impact'] = {
                    'portfolio_risk_score': float(results_df['default_probability'].mean()),
                    'estimated_defaults': int(len(results_df) * results_df['default_probability'].mean()),
                    'high_risk_loans_requiring_review': len(results_df[results_df['risk_category'].isin(['HIGH', 'VERY_HIGH'])]),
                    'recommendations': [
                        'Review all VERY_HIGH risk loans before approval',
                        'Consider additional documentation for HIGH risk loans',
                        'Implement enhanced monitoring for loans with >70% default probability'
                    ]
                }
            
            # Save comprehensive report
            with open(f'{results_path}/batch_inference_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info("Batch inference report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise

    def monitor_batch_job_status(self, job_name, max_wait_time=3600):
        """Monitor batch job status with timeout"""
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                # Check job status (implementation depends on job type)
                # This is a placeholder for job status monitoring
                job_status = self._get_job_status(job_name)
                
                if job_status in ['Completed', 'Succeeded']:
                    logger.info(f"Batch job {job_name} completed successfully")
                    return 'COMPLETED'
                elif job_status in ['Failed', 'Stopped']:
                    logger.error(f"Batch job {job_name} failed")
                    return 'FAILED'
                else:
                    logger.info(f"Batch job {job_name} status: {job_status}")
                    time.sleep(60)  # Check every minute
                    
            except Exception as e:
                logger.error(f"Error monitoring job {job_name}: {str(e)}")
                time.sleep(60)
        
        logger.warning(f"Batch job {job_name} monitoring timed out")
        return 'TIMEOUT'

    def _get_job_status(self, job_name):
        """Get job status from SageMaker"""
        try:
            sm_client = self.boto_session.client('sagemaker')
            
            # Try to get processing job status
            try:
                response = sm_client.describe_processing_job(ProcessingJobName=job_name)
                return response['ProcessingJobStatus']
            except:
                pass
            
            # Try to get transform job status
            try:
                response = sm_client.describe_transform_job(TransformJobName=job_name)
                return response['TransformJobStatus']
            except:
                pass
            
            return 'Unknown'
            
        except Exception as e:
            logger.error(f"Error getting job status: {str(e)}")
            return 'Error'

# Example usage and configuration
def create_batch_inference_example():
    """Example of how to use the batch inference pipeline"""
    
    # Configuration
    config = {
        'role_arn': 'arn:aws:iam::123456789012:role/SageMakerExecutionRole',
        'bucket_name': 'mortgage-ml-bucket',
        'model_data_path': 's3://mortgage-ml-bucket/model/model.tar.gz',
        'region': 'us-east-1'
    }
    
    # Initialize pipeline
    pipeline = MortgageBatchInferencePipeline(**config)
    
    # Example batch inference workflow
    input_data_path = 's3://mortgage-ml-bucket/batch-data/loans_to_predict.csv'
    
    try:
        # Run batch inference
        results_path = pipeline.run_batch_inference_processing(input_data_path)
        
        # Validate results
        validation_results = pipeline.validate_batch_results(results_path)
        
        # Generate report
        report = pipeline.generate_batch_report(results_path)
        
        print(f"Batch inference completed successfully!")
        print(f"Results: {results_path}")
        print(f"Validation: {validation_results['overall_status']}")
        
    except Exception as e:
        print(f"Batch inference failed: {str(e)}")

if __name__ == "__main__":
    create_batch_inference_example()