import boto3
from datetime import datetime

# Replace with your actual AWS Batch Job ID or Array Job ID (e.g., '12345678-1234-1234-1234-1234567890ab')
job_id = "YOUR_BATCH_JOB_ID"
client = boto3.client("batch", region_name="us-west-2")

response = client.describe_jobs(jobs=[job_id])
jobs = response.get("jobs", [])

if not jobs:
    print(f"Batch Job ID {job_id} not found.")
else:
    job = jobs[0]
    print(f"Job Name:   {job.get('jobName')}")
    print(f"Status:     {job.get('status')}")
    
    container = job.get("container", {})
    vcpus = container.get("vcpus", 0)
    memory = container.get("memory", 0)
    
    # AWS Batch execution timestamps are in millisecond epochs
    start_time = container.get("startedAt")
    stop_time = container.get("stoppedAt")
    
    if start_time and stop_time:
        duration_seconds = (stop_time - start_time) / 1000.0
    else:
        duration_seconds = 0
        
    # Assuming Fargate or EC2 pricing model per vCPU-second 
    # (Using standard On-Demand Fargate baseline as an example: ~$0.00001124 per vCPU-second)
    cost_per_vcpu_second = 0.04048 / 3600
    estimated_cost = duration_seconds * vcpus * cost_per_vcpu_second

    print(f"Allocated vCPUs: {vcpus}")
    print(f"Duration:        {duration_seconds:.2f} seconds")
    print(f"Est. Cost:       ${estimated_cost:.6f} USD")