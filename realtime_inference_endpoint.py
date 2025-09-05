import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MortgageRiskInferenceEndpoint:
    
    def __init__(self, role_arn, model_package_arn=None, model_data_path=None, region='us-east-1'):
        """
        Initialize real-time inference endpoint for mortgage risk model
        
        Args:
            role_arn: AWS IAM role ARN with SageMaker permissions
            model_package_arn: ARN of registered model package (optional)
            model_data_path: S3 path to model artifacts (optional)
            region: AWS region
        """
        self.role_arn = role_arn
        self.model_package_arn = model_package_arn
        self.model_data_path = model_data_path
        self.region = region
        
        # Initialize SageMaker session
        self.sagemaker_session = sagemaker.Session()
        self.boto_session = boto3.Session(region_name=region)
        
        logger.info("Initialized MortgageRiskInferenceEndpoint")

    def create_inference_script(self):
        """Create inference script for SageMaker endpoint"""
        
        inference_script = '''
import joblib
import pandas as pd
import numpy as np
import os
import json
from io import StringIO

def model_fn(model_dir):
    """Load model and preprocessing artifacts"""
    
    # Load model artifacts
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    label_encoders = joblib.load(os.path.join(model_dir, "label_encoders.joblib"))
    feature_columns = joblib.load(os.path.join(model_dir, "feature_columns.joblib"))
    
    return {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_columns': feature_columns
    }

def input_fn(request_body, request_content_type):
    """Parse input data"""
    
    if request_content_type == 'text/csv':
        # Parse CSV input
        df = pd.read_csv(StringIO(request_body))
        return df
    elif request_content_type == 'application/json':
        # Parse JSON input
        data = json.loads(request_body)
        if isinstance(data, dict):
            # Single record
            df = pd.DataFrame([data])
        else:
            # Multiple records
            df = pd.DataFrame(data)
        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_artifacts):
    """Make predictions"""
    
    # Extract artifacts
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    label_encoders = model_artifacts['label_encoders']
    feature_columns = model_artifacts['feature_columns']
    
    # Feature engineering (same as training)
    df = feature_engineering(input_data.copy())
    
    # Prepare features
    X = prepare_features_for_inference(df, label_encoders, feature_columns)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    # Prepare response
    results = []
    for i in range(len(X)):
        result = {
            'loan_id': str(input_data.iloc[i].get('loan_id', f'loan_{i}')),
            'default_probability': float(probabilities[i][1]),
            'risk_prediction': int(predictions[i]),
            'risk_category': get_risk_category(probabilities[i][1]),
            'confidence_score': float(max(probabilities[i])),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        results.append(result)
    
    return results

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
    current_year = pd.Timestamp.now().year
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
                # Use most frequent class as fallback
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

def output_fn(prediction, accept):
    """Format prediction output"""
    
    if accept == "application/json":
        return json.dumps(prediction), accept
    elif accept == 'text/csv':
        # Convert to CSV format
        df = pd.DataFrame(prediction)
        return df.to_csv(index=False), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
'''
        
        # Save inference script
        with open('inference.py', 'w') as f:
            f.write(inference_script)
        
        logger.info("Inference script created")
        return 'inference.py'

    def create_model(self, endpoint_name_prefix="mortgage-risk"):
        """Create SageMaker model for deployment"""
        
        # Create inference script
        inference_script_path = self.create_inference_script()
        
        if self.model_package_arn:
            # Use model package
            model = sagemaker.model.ModelPackage(
                role=self.role_arn,
                model_package_arn=self.model_package_arn,
                sagemaker_session=self.sagemaker_session
            )
            
        else:
            # Use model artifacts directly
            model = SKLearnModel(
                model_data=self.model_data_path,
                role=self.role_arn,
                entry_point=inference_script_path,
                framework_version='1.2-1',
                py_version='py3',
                sagemaker_session=self.sagemaker_session,
                name=f"{endpoint_name_prefix}-model-{int(datetime.now().timestamp())}"
            )
        
        logger.info("SageMaker model created")
        return model

    def deploy_endpoint(self, instance_type='ml.t2.medium', instance_count=1, 
                       endpoint_name=None, auto_scaling_config=None):
        """Deploy real-time inference endpoint"""
        
        if not endpoint_name:
            endpoint_name = f"mortgage-risk-endpoint-{int(datetime.now().timestamp())}"
        
        # Create model
        model = self.create_model()
        
        # Deploy endpoint
        predictor = model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            serializer=CSVSerializer(),
            deserializer=JSONDeserializer(),
            wait=True
        )
        
        # Configure auto scaling if provided
        if auto_scaling_config:
            self.configure_auto_scaling(endpoint_name, auto_scaling_config)
        
        logger.info(f"Endpoint deployed: {endpoint_name}")
        self.endpoint_name = endpoint_name
        self.predictor = predictor
        
        return predictor

    def configure_auto_scaling(self, endpoint_name, auto_scaling_config):
        """Configure auto scaling for the endpoint"""
        
        application_autoscaling = self.boto_session.client('application-autoscaling')
        
        # Register scalable target
        application_autoscaling.register_scalable_target(
            ServiceNamespace='sagemaker',
            ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            MinCapacity=auto_scaling_config.get('min_capacity', 1),
            MaxCapacity=auto_scaling_config.get('max_capacity', 10),
            RoleArn=self.role_arn
        )
        
        # Configure scaling policy
        application_autoscaling.put_scaling_policy(
            PolicyName=f'{endpoint_name}-scaling-policy',
            ServiceNamespace='sagemaker',
            ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            PolicyType='TargetTrackingScaling',
            TargetTrackingScalingPolicyConfiguration={
                'TargetValue': auto_scaling_config.get('target_cpu_utilization', 70.0),
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                },
                'ScaleOutCooldown': auto_scaling_config.get('scale_out_cooldown', 300),
                'ScaleInCooldown': auto_scaling_config.get('scale_in_cooldown', 300)
            }
        )
        
        logger.info(f"Auto scaling configured for endpoint: {endpoint_name}")

    def test_endpoint(self, test_data_sample):
        """Test the deployed endpoint with sample data"""
        
        if not hasattr(self, 'predictor'):
            raise ValueError("Endpoint not deployed. Call deploy_endpoint() first.")
        
        # Convert test data to CSV format
        if isinstance(test_data_sample, dict):
            df = pd.DataFrame([test_data_sample])
        else:
            df = pd.DataFrame(test_data_sample)
        
        csv_data = df.to_csv(index=False)
        
        # Make prediction
        try:
            result = self.predictor.predict(csv_data)
            logger.info("Endpoint test successful")
            return result
        
        except Exception as e:
            logger.error(f"Endpoint test failed: {str(e)}")
            raise

    def create_endpoint_with_traffic_routing(self, models_config, endpoint_name=None):
        """Create endpoint with traffic routing between multiple models"""
        
        if not endpoint_name:
            endpoint_name = f"mortgage-risk-multi-model-{int(datetime.now().timestamp())}"
        
        # Create endpoint configuration
        sagemaker_client = self.boto_session.client('sagemaker')
        
        # Prepare production variants
        production_variants = []
        
        for i, model_config in enumerate(models_config):
            model = self.create_model(f"mortgage-risk-model-{i}")
            
            variant = {
                'VariantName': model_config.get('variant_name', f'variant-{i}'),
                'ModelName': model.name,
                'InitialInstanceCount': model_config.get('instance_count', 1),
                'InstanceType': model_config.get('instance_type', 'ml.t2.medium'),
                'InitialVariantWeight': model_config.get('traffic_weight', 1.0)
            }
            production_variants.append(variant)
        
        # Create endpoint configuration
        config_name = f"{endpoint_name}-config"
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=production_variants
        )
        
        # Create endpoint
        sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        
        # Wait for endpoint to be in service
        waiter = sagemaker_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)
        
        logger.info(f"Multi-model endpoint created: {endpoint_name}")
        return endpoint_name

    def update_endpoint_traffic(self, endpoint_name, new_weights):
        """Update traffic weights for A/B testing"""
        
        sagemaker_client = self.boto_session.client('sagemaker')
        
        # Get current endpoint config
        endpoint_desc = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        current_config = endpoint_desc['EndpointConfigName']
        
        # Create new endpoint config with updated weights
        config_desc = sagemaker_client.describe_endpoint_config(
            EndpointConfigName=current_config
        )
        
        # Update weights
        variants = config_desc['ProductionVariants']
        for variant in variants:
            variant_name = variant['VariantName']
            if variant_name in new_weights:
                variant['InitialVariantWeight'] = new_weights[variant_name]
        
        # Create new config
        new_config_name = f"{current_config}-{int(datetime.now().timestamp())}"
        sagemaker_client.create_endpoint_config(
            EndpointConfigName=new_config_name,
            ProductionVariants=variants
        )
        
        # Update endpoint
        sagemaker_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=new_config_name
        )
        
        logger.info(f"Endpoint traffic weights updated: {endpoint_name}")

    def cleanup_endpoint(self, endpoint_name=None):
        """Delete endpoint and related resources"""
        
        if endpoint_name is None and hasattr(self, 'endpoint_name'):
            endpoint_name = self.endpoint_name
        
        if endpoint_name is None:
            logger.warning("No endpoint name provided for cleanup")
            return
        
        sagemaker_client = self.boto_session.client('sagemaker')
        
        try:
            # Delete endpoint
            sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"Endpoint deleted: {endpoint_name}")
            
            # Optionally delete endpoint config and model
            # (commented out for safety)
            # sagemaker_client.delete_endpoint_config(EndpointConfigName=config_name)
            # sagemaker_client.delete_model(ModelName=model_name)
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

def main():
    """Example usage"""
    
    # Configuration
    ROLE_ARN = 'arn:aws:iam::123456789012:role/SageMakerExecutionRole'
    MODEL_PACKAGE_ARN = 'arn:aws:sagemaker:us-east-1:123456789012:model-package/mortgage-risk-model-group/1'
    
    # Initialize endpoint manager
    endpoint_manager = MortgageRiskInferenceEndpoint(
        role_arn=ROLE_ARN,
        model_package_arn=MODEL_PACKAGE_ARN
    )
    
    # Deploy endpoint with auto scaling
    auto_scaling_config = {
        'min_capacity': 1,
        'max_capacity': 5,
        'target_cpu_utilization': 70.0
    }
    
    predictor = endpoint_manager.deploy_endpoint(
        instance_type='ml.m5.large',
        instance_count=1,
        auto_scaling_config=auto_scaling_config
    )
    
    # Test endpoint
    sample_loan = {
        'loan_id': 'TEST001',
        'loan_amount': 250000,
        'property_value': 300000,
        'monthly_income': 8000,
        'monthly_debt': 2000,
        'credit_score': 720,
        'loan_term': 30,
        'interest_rate': 3.5,
        'down_payment': 50000,
        'credit_used': 15000,
        'credit_limit': 25000,
        'monthly_payment': 1123,
        'birth_year': 1985,
        'year_built': 2010,
        'primary_credit_score': 720,
        'secondary_credit_score': 715,
        'loan_purpose': 'purchase',
        'property_type': 'single_family',
        'occupancy_type': 'primary',
        'employment_type': 'full_time',
        'state': 'CA',
        'loan_type': 'conventional'
    }
    
    # Make prediction
    result = endpoint_manager.test_endpoint(sample_loan)
    print("Prediction Result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
