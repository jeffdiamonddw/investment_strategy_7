import argparse
import boto3
import sys

def main():
    parser = argparse.ArgumentParser()
    # ... (Keep your existing arguments)
    parser.add_argument('--s3_path', required=True)
    parser.add_argument('--generation', type=int, required=True)
    parser.add_argument('--train_folds', type=int, nargs='+', default=[])
    parser.add_argument('--val_folds', type=int, nargs='+', default=[])
    parser.add_argument('--array_size', type=int, default=2)
    args = parser.parse_args()

    batch = boto3.client('batch', region_name='us-west-2')
    job_def_name = "simulation-job-def"

    # 1. Define resources within the script
    # This replaces the need for the external job-definition.json
    print(f"Registering/Updating Job Definition: {job_def_name}...")
    batch.register_job_definition(
        jobDefinitionName=job_def_name,
        type='container',
        containerProperties={
            'image': '129861351772.dkr.ecr.us-west-2.amazonaws.com/simulation:latest',
            'vcpus': 1,
            'memory': 1024,
            'jobRoleArn': 'arn:aws:iam::129861351772:role/ecsTaskExecutionRole',
            'executionRoleArn': 'arn:aws:iam::129861351772:role/ecsTaskExecutionRole',
            
        }
    )

    # 2. Submit the job
    cmd = [
        "python3", "python_scripts/regime_navigator_2d.py",
        "--s3_path", args.s3_path,
        "--generation", str(args.generation),
        "--train_folds", *map(str, args.train_folds),
        "--val_folds", *map(str, args.val_folds)
    ]

    print("Submitting array job...")
    response = batch.submit_job(
        jobName=f"sim-gen-{args.generation}",
        jobQueue="batch-arm-192-queue",
        jobDefinition=job_def_name,
        arrayProperties={'size': args.array_size},
        containerOverrides={'command': cmd}
    )

    print(f"Successfully submitted! Job ID: {response['jobId']}")

if __name__ == "__main__":
    main()