import boto3

def get_array_status_summary(parent_job_id):
    batch = boto3.client('batch')
    
    response = batch.describe_jobs(jobs=[parent_job_id])
    
    if not response['jobs']:
        return "Job not found."
    
    job_detail = response['jobs'][0]
    
    # Check if it's actually an array job
    if 'arrayProperties' in job_detail and 'statusSummary' in job_detail['arrayProperties']:
        return job_detail['arrayProperties']['statusSummary']
    else:
        return "No status summary found (it might not be a parent array job)."


def job_complete(parent_job_id):
    summary = get_array_status_summary('4b3886f7-6a23-4c86-a829-d7e34de7ab0b')
    done_count = summary['SUCCEEDED'] + summary['FAILED']
    total_count = sum(list(summary.values()))
    return done_count == total_count


# Usage
print(job_complete('4b3886f7-6a23-4c86-a829-d7e34de7ab0b'))
