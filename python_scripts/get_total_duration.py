import boto3
import sys

def get_array_job_total_duration(array_job_id):
    batch = boto3.client('batch', region_name='us-west-2')
    
    # Use a paginator to ensure we get all child jobs
    paginator = batch.get_paginator('list_jobs')
    
    # By passing jobStatus=None (or leaving it out), you only get RUNNING jobs.
    # To get finished jobs, you must either specify the status or 
    # check the documentation for filtering behavior.
    # Here, we fetch all statuses to be safe:
    child_job_ids = []
    for status in ['SUCCEEDED', 'FAILED', 'RUNNING', 'RUNNABLE', 'PENDING']:
        for page in paginator.paginate(arrayJobId=array_job_id, jobStatus=status):
            for job in page['jobSummaryList']:
                child_job_ids.append(job['jobId'])

    if not child_job_ids:
        print(f"No child jobs found for array ID: {array_job_id}")
        return

    # Describe jobs to get timestamps
    total_duration_ms = 0
    # Process in batches of 100 as required by DescribeJobs API
    for i in range(0, len(child_job_ids), 100):
        batch_ids = child_job_ids[i:i+100]
        response = batch.describe_jobs(jobs=batch_ids)
        
        for job in response['jobs']:
            # Only include duration if the job actually started and stopped
            if 'startedAt' in job and 'stoppedAt' in job:
                duration = job['stoppedAt'] - job['startedAt']
                total_duration_ms += duration
                print(f"Child {job['jobId']} duration: {duration/1000:.2f}s")

    print(f"---")
    print(f"Total cumulative duration: {total_duration_ms/1000:.2f} seconds")

if __name__ == "__main__":
    get_array_job_total_duration(sys.argv[1])