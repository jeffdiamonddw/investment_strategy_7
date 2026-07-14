#!/bin/bash

# Configuration
QUEUE_NAME="batch-arm-192-queue"
REGION="us-west-2"

echo "--- Examining Batch Queue: $QUEUE_NAME ---"

# 1. Describe the Job Queue
aws batch describe-job-queues \
    --job-queues "$QUEUE_NAME" \
    --region "$REGION"

# 2. Extract and Describe the associated Compute Environment
# This helps us see if the CE is ENABLED and what instances it can spin up
CE_NAME=$(aws batch describe-job-queues --job-queues "$QUEUE_NAME" --query "jobQueues[0].computeEnvironmentOrder[0].computeEnvironment" --output text --region "$REGION")

echo -e "\n--- Examining Compute Environment: $CE_NAME ---"
aws batch describe-compute-environments \
    --compute-environments "$CE_NAME" \
    --region "$REGION"