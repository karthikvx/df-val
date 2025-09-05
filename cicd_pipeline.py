import boto3
import json
import yaml
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLCICDPipeline:
    
    def __init__(self, pipeline_name, region='us-east-1', 
                 source_repo=None, artifact_bucket=None):
        """
        Initialize ML CI/CD Pipeline
        
        Args:
            pipeline_name: Name of the CI/CD pipeline
            region: AWS region
            source_repo: Source repository configuration
            artifact_bucket: S3 bucket for pipeline artifacts
        """
        self.pipeline_name = pipeline_name
        self.region = region
        self.source_repo = source_repo
        self.artifact_bucket = artifact_bucket
        
        # Initialize AWS clients
        self.codepipeline = boto3.client('codepipeline', region_name=region)
        self.codebuild = boto3.client('codebuild', region_name=region)
        self.iam = boto3.client('iam', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        
        logger.info(f"Initialized ML CI/CD Pipeline: {pipeline_name}")

    def create_service_role(self, role_name, service_principal, policies):
        """Create IAM service role for pipeline components"""
        
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": service_principal},
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        try:
            # Create role
            response = self.iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description=f"Service role for {role_name}"
            )
            
            # Attach policies
            for policy_arn in policies:
                self.iam.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy_arn
                )
            
            role_arn = response['Role']['Arn']
            logger.info(f"Created service role: {role_arn}")
            return role_arn
            
        except Exception as e:
            if 'EntityAlreadyExists' in str(e):
                response = self.iam.get_role(RoleName=role_name)
                return response['Role']['Arn']
            else:
                logger.error(f"Failed to create role {role_name}: {str(e)}")
                raise

    def create_codebuild_projects(self):
        """Create CodeBuild projects for different pipeline stages"""
        
        projects = {}
        
        # 1. Data Validation and Preprocessing Project
        data_validation_spec = {
            'version': '0.2',
            'phases': {
                'install': {
                    'runtime-versions': {'python': '3.9'},
                    'commands': [
                        'pip install --upgrade pip',
                        'pip install pandas numpy scikit-learn boto3 sagemaker great-expectations'
                    ]
                },
                'pre_build': {
                    'commands': [
                        'echo "Starting data validation phase"',
                        'export PYTHONPATH=$PYTHONPATH:$CODEBUILD_SRC_DIR'
                    ]
                },
                'build': {
                    'commands': [
                        'python scripts/validate_data.py --input-path $DATA_INPUT_PATH --output-path $DATA_OUTPUT_PATH',
                        'python scripts/data_preprocessing.py --input-path $DATA_INPUT_PATH --output-path $PREPROCESSED_DATA_PATH',
                        'python scripts/generate_data_profile.py --data-path $PREPROCESSED_DATA_PATH --profile-output $DATA_PROFILE_PATH'
                    ]
                },
                'post_build': {
                    'commands': [
                        'echo "Data validation completed"',
                        'aws s3 cp data_validation_report.json s3://$ARTIFACT_BUCKET/reports/',
                        'aws s3 cp data_profile.json s3://$ARTIFACT_BUCKET/reports/'
                    ]
                }
            },
            'artifacts': {
                'files': [
                    'data_validation_report.json',
                    'data_profile.json',
                    'preprocessed_data/**/*'
                ]
            },
            'reports': {
                'data-validation-report': {
                    'files': ['data_validation_report.json'],
                    'file-format': 'JUNITXML'
                }
            }
        }
        
        projects['data-validation'] = self.create_build_project(
            f"{self.pipeline_name}-data-validation",
            data_validation_spec,
            'BUILD_GENERAL1_MEDIUM'
        )
        
        # 2. Model Training Project
        training_spec = {
            'version': '0.2',
            'phases': {
                'install': {
                    'runtime-versions': {'python': '3.9'},
                    'commands': [
                        'pip install --upgrade pip',
                        'pip install pandas numpy scikit-learn boto3 sagemaker joblib',
                        'pip install xgboost lightgbm optuna mlflow'
                    ]
                },
                'pre_build': {
                    'commands': [
                        'echo "Starting model training phase"',
                        'export PYTHONPATH=$PYTHONPATH:$CODEBUILD_SRC_DIR',
                        'mkdir -p model_artifacts'
                    ]
                },
                'build': {
                    'commands': [
                        'python scripts/train_model.py --data-path $PREPROCESSED_DATA_PATH --model-output model_artifacts/',
                        'python scripts/model_evaluation.py --model-path model_artifacts/ --test-data $TEST_DATA_PATH',
                        'python scripts/generate_model_report.py --model-path model_artifacts/ --report-output model_report.json'
                    ]
                },
                'post_build': {
                    'commands': [
                        'echo "Model training completed"',
                        'tar -czf model.tar.gz -C model_artifacts .',
                        'aws s3 cp model.tar.gz s3://$ARTIFACT_BUCKET/models/$CODEBUILD_BUILD_NUMBER/',
                        'aws s3 cp model_report.json s3://$ARTIFACT_BUCKET/reports/',
                        'echo "Model uploaded to S3"'
                    ]
                }
            },
            'artifacts': {
                'files': [
                    'model.tar.gz',
                    'model_report.json',
                    'model_metrics.json'
                ]
            },
            'reports': {
                'model-training-report': {
                    'files': ['model_report.json'],
                    'file-format': 'JUNITXML'
                }
            }
        }
        
        projects['model-training'] = self.create_build_project(
            f"{self.pipeline_name}-training",
            training_spec,
            'BUILD_GENERAL1_LARGE'
        )
        
        # 3. Model Testing and Validation Project
        testing_spec = {
            'version': '0.2',
            'phases': {
                'install': {
                    'runtime-versions': {'python': '3.9'},
                    'commands': [
                        'pip install --upgrade pip',
                        'pip install pandas numpy scikit-learn boto3 sagemaker pytest',
                        'pip install evidently alibi-detect'
                    ]
                },
                'pre_build': {
                    'commands': [
                        'echo "Starting model testing phase"',
                        'aws s3 cp s3://$ARTIFACT_BUCKET/models/$CODEBUILD_BUILD_NUMBER/model.tar.gz .',
                        'tar -xzf model.tar.gz -C model_artifacts/'
                    ]
                },
                'build': {
                    'commands': [
                        'python -m pytest tests/test_model_functionality.py -v --junitxml=functionality_tests.xml',
                        'python scripts/model_performance_validation.py --model-path model_artifacts/ --test-data $TEST_DATA_PATH',
                        'python scripts/bias_fairness_testing.py --model-path model_artifacts/ --test-data $TEST_DATA_PATH',
                        'python scripts/model_stability_testing.py --model-path model_artifacts/ --test-data $TEST_DATA_PATH'
                    ]
                },
                'post_build': {
                    'commands': [
                        'echo "Model testing completed"',
                        'aws s3 cp test_results.json s3://$ARTIFACT_BUCKET/reports/',
                        'aws s3 cp bias_report.json s3://$ARTIFACT_BUCKET/reports/',
                        'aws s3 cp stability_report.json s3://$ARTIFACT_BUCKET/reports/'
                    ]
                }
            },
            'artifacts': {
                'files': [
                    'test_results.json',
                    'bias_report.json',
                    'stability_report.json',
                    'functionality_tests.xml'
                ]
            },
            'reports': {
                'model-tests': {
                    'files': ['functionality_tests.xml'],
                    'file-format': 'JUNITXML'
                }
            }
        }
        
        projects['model-testing'] = self.create_build_project(
            f"{self.pipeline_name}-testing",
            testing_spec,
            'BUILD_GENERAL1_MEDIUM'
        )
        
        # 4. Model Deployment to Staging Project
        staging_deploy_spec = {
            'version': '0.2',
            'phases': {
                'install': {
                    'runtime-versions': {'python': '3.9'},
                    'commands': [
                        'pip install --upgrade pip',
                        'pip install boto3 sagemaker'
                    ]
                },
                'pre_build': {
                    'commands': [
                        'echo "Starting staging deployment"',
                        'aws s3 cp s3://$ARTIFACT_BUCKET/models/$CODEBUILD_BUILD_NUMBER/model.tar.gz .'
                    ]
                },
                'build': {
                    'commands': [
                        'python scripts/deploy_to_staging.py --model-path model.tar.gz --endpoint-name $STAGING_ENDPOINT_NAME',
                        'python scripts/staging_integration_tests.py --endpoint-name $STAGING_ENDPOINT_NAME',
                        'python scripts/load_testing.py --endpoint-name $STAGING_ENDPOINT_NAME --test-duration 300'
                    ]
                },
                'post_build': {
                    'commands': [
                        'echo "Staging deployment completed"',
                        'aws s3 cp staging_test_results.json s3://$ARTIFACT_BUCKET/reports/',
                        'aws s3 cp load_test_results.json s3://$ARTIFACT_BUCKET/reports/'
                    ]
                }
            },
            'artifacts': {
                'files': [
                    'staging_test_results.json',
                    'load_test_results.json'
                ]
            }
        }
        
        projects['staging-deploy'] = self.create_build_project(
            f"{self.pipeline_name}-staging-deploy",
            staging_deploy_spec,
            'BUILD_GENERAL1_MEDIUM'
        )
        
        # 5. Production Deployment Project
        prod_deploy_spec = {
            'version': '0.2',
            'phases': {
                'install': {
                    'runtime-versions': {'python': '3.9'},
                    'commands': [
                        'pip install --upgrade pip',
                        'pip install boto3 sagemaker'
                    ]
                },
                'pre_build': {
                    'commands': [
                        'echo "Starting production deployment"',
                        'aws s3 cp s3://$ARTIFACT_BUCKET/models/$CODEBUILD_BUILD_NUMBER/model.tar.gz .'
                    ]
                },
                'build': {
                    'commands': [
                        'python scripts/deploy_to_production.py --model-path model.tar.gz --endpoint-name $PROD_ENDPOINT_NAME --deployment-strategy blue-green',
                        'python scripts/production_smoke_tests.py --endpoint-name $PROD_ENDPOINT_NAME',
                        'python scripts/register_model.py --model-path model.tar.gz --model-package-group $MODEL_PACKAGE_GROUP'
                    ]
                },
                'post_build': {
                    'commands': [
                        'echo "Production deployment completed"',
                        'aws s3 cp production_deployment_report.json s3://$ARTIFACT_BUCKET/reports/'
                    ]
                }
            },
            'artifacts': {
                'files': ['production_deployment_report.json']
            }
        }
        
        projects['production-deploy'] = self.create_build_project(
            f"{self.pipeline_name}-production-deploy",
            prod_deploy_spec,
            'BUILD_GENERAL1_MEDIUM'
        )
        
        logger.info("Created all CodeBuild projects")
        return projects

    def create_build_project(self, project_name, buildspec, compute_type):
        """Create individual CodeBuild project"""
        
        try:
            # Create service role for CodeBuild
            role_arn = self.create_service_role(
                f"{project_name}-role",
                "codebuild.amazonaws.com",
                [
                    "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess",
                    "arn:aws:iam::aws:policy/AmazonS3FullAccess",
                    "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
                ]
            )
            
            # Create CodeBuild project
            response = self.codebuild.create_project(
                name=project_name,
                description=f"CodeBuild project for {project_name}",
                source={
                    'type': 'CODEPIPELINE',
                    'buildspec': yaml.dump(buildspec)
                },
                artifacts={'type': 'CODEPIPELINE'},
                environment={
                    'type': 'LINUX_CONTAINER',
                    'image': 'aws/codebuild/standard:5.0',
                    'computeType': compute_type,
                    'environmentVariables': [
                        {
                            'name': 'ARTIFACT_BUCKET',
                            'value': self.artifact_bucket
                        },
                        {
                            'name': 'AWS_DEFAULT_REGION',
                            'value': self.region
                        }
                    ]
                },
                serviceRole=role_arn,
                timeoutInMinutes=60
            )
            
            logger.info(f"Created CodeBuild project: {project_name}")
            return response['project']['arn']
            
        except Exception as e:
            if 'ResourceAlreadyExistsException' in str(e):
                logger.info(f"CodeBuild project {project_name} already exists")
                return f"arn:aws:codebuild:{self.region}:*:project/{project_name}"
            else:
                logger.error(f"Failed to create CodeBuild project {project_name}: {str(e)}")
                raise

    def create_pipeline(self, projects):
        """Create the main CI/CD pipeline"""
        
        try:
            # Create service role for CodePipeline
            pipeline_role_arn = self.create_service_role(
                f"{self.pipeline_name}-pipeline-role",
                "codepipeline.amazonaws.com",
                [
                    "arn:aws:iam::aws:policy/AWSCodePipelineFullAccess",
                    "arn:aws:iam::aws:policy/AWSCodeBuildDeveloperAccess",
                    "arn:aws:iam::aws:policy/AmazonS3FullAccess"
                ]
            )
            
            # Define pipeline structure
            pipeline_definition = {
                'name': self.pipeline_name,
                'roleArn': pipeline_role_arn,
                'artifactStore': {
                    'type': 'S3',
                    'location': self.artifact_bucket
                },
                'stages': [
                    # Source Stage
                    {
                        'name': 'Source',
                        'actions': [
                            {
                                'name': 'SourceAction',
                                'actionTypeId': {
                                    'category': 'Source',
                                    'owner': 'AWS',
                                    'provider': 'S3',
                                    'version': '1'
                                },
                                'configuration': {
                                    'S3Bucket': self.source_repo['bucket'],
                                    'S3ObjectKey': self.source_repo['key']
                                },
                                'outputArtifacts': [{'name': 'SourceOutput'}]
                            }
                        ]
                    },
                    
                    # Data Validation Stage
                    {
                        'name': 'DataValidation',
                        'actions': [
                            {
                                'name': 'ValidateData',
                                'actionTypeId': {
                                    'category': 'Build',
                                    'owner': 'AWS',
                                    'provider': 'CodeBuild',
                                    'version': '1'
                                },
                                'configuration': {
                                    'ProjectName': f"{self.pipeline_name}-data-validation"
                                },
                                'inputArtifacts': [{'name': 'SourceOutput'}],
                                'outputArtifacts': [{'name': 'DataValidationOutput'}]
                            }
                        ]
                    },
                    
                    # Model Training Stage
                    {
                        'name': 'ModelTraining',
                        'actions': [
                            {
                                'name': 'TrainModel',
                                'actionTypeId': {
                                    'category': 'Build',
                                    'owner': 'AWS',
                                    'provider': 'CodeBuild',
                                    'version': '1'
                                },
                                'configuration': {
                                    'ProjectName': f"{self.pipeline_name}-training"
                                },
                                'inputArtifacts': [
                                    {'name': 'SourceOutput'},
                                    {'name': 'DataValidationOutput'}
                                ],
                                'outputArtifacts': [{'name': 'TrainingOutput'}]
                            }
                        ]
                    },
                    
                    # Model Testing Stage
                    {
                        'name': 'ModelTesting',
                        'actions': [
                            {
                                'name': 'TestModel',
                                'actionTypeId': {
                                    'category': 'Build',
                                    'owner': 'AWS',
                                    'provider': 'CodeBuild',
                                    'version': '1'
                                },
                                'configuration': {
                                    'ProjectName': f"{self.pipeline_name}-testing"
                                },
                                'inputArtifacts': [
                                    {'name': 'SourceOutput'},
                                    {'name': 'TrainingOutput'}
                                ],
                                'outputArtifacts': [{'name': 'TestingOutput'}]
                            }
                        ]
                    },
                    
                    # Staging Deployment Stage
                    {
                        'name': 'StagingDeployment',
                        'actions': [
                            {
                                'name': 'DeployToStaging',
                                'actionTypeId': {
                                    'category': 'Build',
                                    'owner': 'AWS',
                                    'provider': 'CodeBuild',
                                    'version': '1'
                                },
                                'configuration': {
                                    'ProjectName': f"{self.pipeline_name}-staging-deploy"
                                },
                                'inputArtifacts': [
                                    {'name': 'SourceOutput'},
                                    {'name': 'TrainingOutput'}
                                ],
                                'outputArtifacts': [{'name': 'StagingOutput'}]
                            }
                        ]
                    },
                    
                    # Manual Approval Stage
                    {
                        'name': 'ProductionApproval',
                        'actions': [
                            {
                                'name': 'ManualApproval',
                                'actionTypeId': {
                                    'category': 'Approval',
                                    'owner': 'AWS',
                                    'provider': 'Manual',
                                    'version': '1'
                                },
                                'configuration': {
                                    'CustomData': 'Please review staging test results and approve for production deployment'
                                }
                            }
                        ]
                    },
                    
                    # Production Deployment Stage
                    {
                        'name': 'ProductionDeployment',
                        'actions': [
                            {
                                'name': 'DeployToProduction',
                                'actionTypeId': {
                                    'category': 'Build',
                                    'owner': 'AWS',
                                    'provider': 'CodeBuild',
                                    'version': '1'
                                },
                                'configuration': {
                                    'ProjectName': f"{self.pipeline_name}-production-deploy"
                                },
                                'inputArtifacts': [
                                    {'name': 'SourceOutput'},
                                    {'name': 'TrainingOutput'}
                                ],
                                'outputArtifacts': [{'name': 'ProductionOutput'}]
                            }
                        ]
                    }
                ]
            }
            
            # Create pipeline
            response = self.codepipeline.create_pipeline(pipeline=pipeline_definition)
            
            logger.info(f"Created CI/CD pipeline: {self.pipeline_name}")
            return response['pipeline']['name']
            
        except Exception as e:
            logger.error(f"Failed to create pipeline: {str(e)}")
            raise

    def create_pipeline_notifications(self, sns_topic_arn):
        """Create CloudWatch Events for pipeline notifications"""
        
        events_client = boto3.client('events', region_name=self.region)
        
        # Pipeline state change rule
        rule_name = f"{self.pipeline_name}-state-change"
        
        try:
            events_client.put_rule(
                Name=rule_name,
                EventPattern=json.dumps({
                    "source": ["aws.codepipeline"],
                    "detail-type": ["CodePipeline Pipeline Execution State Change"],
                    "detail": {
                        "pipeline": [self.pipeline_name],
                        "state": ["FAILED", "SUCCEEDED", "CANCELED"]
                    }
                }),
                State='ENABLED',
                Description=f'Pipeline state changes for {self.pipeline_name}'
            )
            
            # Add SNS target
            events_client.put_targets(
                Rule=rule_name,
                Targets=[
                    {
                        'Id': '1',
                        'Arn': sns_topic_arn,
                        'InputTransformer': {
                            'InputPathsMap': {
                                'pipeline': '$.detail.pipeline',
                                'state': '$.detail.state',
                                'execution-id': '$.detail.execution-id'
                            },
                            'InputTemplate': '''
{
  "pipeline": "<pipeline>",
  "state": "<state>",
  "execution-id": "<execution-id>",
  "timestamp": "$.time"
}'''
                        }
                    }
                ]
            )
            
            logger.info(f"Created pipeline notifications for {self.pipeline_name}")
            
        except Exception as e:
            logger.error(f"Failed to create notifications: {str(e)}")
            raise

    def create_deployment_scripts(self):
        """Create deployment scripts for different stages"""
        
        scripts = {}
        
        # Staging deployment script
        staging_deploy_script = '''
import boto3
import json
import time
import os
from datetime import datetime

def deploy_to_staging(model_path, endpoint_name):
    """Deploy model to staging environment"""
    
    sagemaker = boto3.client('sagemaker')
    
    try:
        # Create model
        model_name = f"{endpoint_name}-model-{int(time.time())}"
        
        response = sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': '246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3',
                'ModelDataUrl': model_path,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
                }
            },
            ExecutionRoleArn=os.environ['SAGEMAKER_EXECUTION_ROLE']
        )
        
        # Create endpoint configuration
        config_name = f"{endpoint_name}-config-{int(time.time())}"
        
        sagemaker.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    'VariantName': 'staging',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.m5.large',
                    'InitialVariantWeight': 1
                }
            ]
        )
        
        # Create or update endpoint
        try:
            sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
        except Exception as e:
            if 'ValidationException' in str(e):
                # Update existing endpoint
                sagemaker.update_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=config_name
                )
        
        # Wait for endpoint to be in service
        waiter = sagemaker.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)
        
        print(f"Successfully deployed to staging: {endpoint_name}")
        return True
        
    except Exception as e:
        print(f"Staging deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1]
    endpoint_name = sys.argv[2]
    deploy_to_staging(model_path, endpoint_name)
'''
        
        scripts['staging_deploy'] = staging_deploy_script
        
        # Production deployment script with blue-green deployment
        production_deploy_script = '''
import boto3
import json
import time
import os
from datetime import datetime

def deploy_to_production(model_path, endpoint_name, deployment_strategy='blue-green'):
    """Deploy model to production with blue-green deployment"""
    
    sagemaker = boto3.client('sagemaker')
    
    try:
        # Create new model
        model_name = f"{endpoint_name}-model-{int(time.time())}"
        
        sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': '246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3',
                'ModelDataUrl': model_path,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'inference.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
                }
            },
            ExecutionRoleArn=os.environ['SAGEMAKER_EXECUTION_ROLE']
        )
        
        if deployment_strategy == 'blue-green':
            # Get current endpoint configuration
            current_endpoint = sagemaker.describe_endpoint(EndpointName=endpoint_name)
            current_config = current_endpoint['EndpointConfigName']
            
            # Create new endpoint configuration
            new_config_name = f"{endpoint_name}-config-{int(time.time())}"
            
            sagemaker.create_endpoint_config(
                EndpointConfigName=new_config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'production',
                        'ModelName': model_name,
                        'InitialInstanceCount': 2,
                        'InstanceType': 'ml.m5.xlarge',
                        'InitialVariantWeight': 1
                    }
                ]
            )
            
            # Update endpoint with new configuration
            sagemaker.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=new_config_name
            )
            
            # Wait for update to complete
            waiter = sagemaker.get_waiter('endpoint_in_service')
            waiter.wait(EndpointName=endpoint_name)
            
            # Cleanup old configuration after successful deployment
            try:
                sagemaker.delete_endpoint_config(EndpointConfigName=current_config)
            except:
                pass  # Don't fail deployment if cleanup fails
        
        print(f"Successfully deployed to production: {endpoint_name}")
        return True
        
    except Exception as e:
        print(f"Production deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1]
    endpoint_name = sys.argv[2]
    strategy = sys.argv[3] if len(sys.argv) > 3 else 'blue-green'
    deploy_to_production(model_path, endpoint_name, strategy)
'''
        
        scripts['production_deploy'] = production_deploy_script
        
        # Model registration script
        model_registration_script = '''
import boto3
import json
import os
from datetime import datetime

def register_model(model_path, model_package_group, approval_status='PendingManualApproval'):
    """Register model in SageMaker Model Registry"""
    
    sagemaker = boto3.client('sagemaker')
    
    try:
        # Get model metadata from environment or defaults
        model_name = os.environ.get('MODEL_NAME', f'mortgage-risk-model-{int(time.time())}')
        model_description = os.environ.get('MODEL_DESCRIPTION', f"Mortgage risk model - {datetime.now().isoformat()}")
        
        # Create model package
        response = sagemaker.create_model_package(
            ModelPackageGroupName=model_package_group,
            ModelPackageDescription=model_description,
            InferenceSpecification={
                'Containers': [
                    {
                        'Image': '246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3',
                        'ModelDataUrl': model_