import boto3
import json

def copy_compute_environment(source_arn, new_name, new_instance_role_arn):
    batch = boto3.client('batch', region_name='us-west-2')
    
    # 1. Describe the existing environment
    response = batch.describe_compute_environments(computeEnvironments=[source_arn])
    source_env = response['computeEnvironments'][0]
    
    # 2. Extract resource configuration
    resources = source_env['computeResources']
    
    # 3. Force Spot and ARM64 configuration
    resources['type'] = 'SPOT'
    resources['instanceRole'] = new_instance_role_arn
    
    # Explicitly set the ARM64 image type for Graviton instances
    # Change the imageType to the valid ECS_AL2023
    resources['ec2Configuration'] = [
        {
            'imageType': 'ECS_AL2023'
        }
    ]
    
    # Ensure optimal instance types are used (which will now be ARM-based)
    resources['instanceTypes'] = ['default_arm64']
    
    # 4. Create the new environment
    print(f"Creating new environment: {new_name} (Spot + ARM64)...")
    create_response = batch.create_compute_environment(
        computeEnvironmentName=new_name,
        type='MANAGED',  # Managed environment is required for Spot
        state='ENABLED',
        computeResources=resources,
        serviceRole=source_env['serviceRole']
    )
    
    print(f"Success! New ARN: {create_response['computeEnvironmentArn']}")

if __name__ == "__main__":
    copy_compute_environment(
        source_arn="arn:aws:batch:us-west-2:129861351772:compute-environment/batch-arm-192-spot",
        new_name="batch-arm-spot-4",
        # Ensure this is the Instance Profile ARN
        new_instance_role_arn="arn:aws:iam::129861351772:instance-profile/batch-arm-instance-role"
    )