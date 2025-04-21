#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3,5
export PET_NODE_RANK=0
# export NumExpr=32
export RAY_DEBUG_POST_MORTEM=1
export NUMEXPR_NUM_THREADS=32
# Default port Ray head node
RAY_PORT=${RAY_PORT:-6379}
RAY_TEMP_DIR="/orange/yonghui.wu/yxliu/tmp" # Define the temp direcdctory

# Explicitly define the GPUs Ray should manage for consistency
# export CUDA_VISIBLE_DEVICES=1,7
# Calculate the number of GPUs based on the comma-separated list
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# Create the temp directory if it doesn't exist
mkdir -p "$RAY_TEMP_DIR"

# Stop any existing Ray cluster on the same port
# ray stop

# Start the Ray head node
echo "Starting Ray head node on port $RAY_PORT using temp dir $RAY_TEMP_DIR with $NUM_GPUS GPUs ($CUDA_VISIBLE_DEVICES)..."
ray start --head \
    --port "$RAY_PORT" \
    --num-gpus "$NUM_GPUS" \
    --dashboard-host 0.0.0.0 \
    --num-cpus 32 \
    --memory "$((256 * 1024 * 1024 * 1024))" \
    --temp-dir "$RAY_TEMP_DIR" # Set the temporary directory

echo "Ray head node started."

# Wait a few seconds for the node to fully register
echo "Waiting 5 seconds for node registration..."
sleep 5

# Check cluster health and resources
echo "--- Running Ray Health Check ---"
ray health-check
echo "--- Health Check Complete ---"

# Specifically check GPU resources using ray status
echo "--- Checking GPU Resources via 'ray status' ---"
ray status | grep 'GPU' || echo "No GPU resources reported by 'ray status' or grep failed."
echo "--- Resource Check Complete ---"

echo "Run 'ray stop' to shut down the cluster."
