
#!/usr/bin/env bash
set -euo pipefail

# Base endpoint name; set USE_SUFFIX=true to create unique endpoints
ENDPOINT_BASE_NAME="mnist-batch"
USE_SUFFIX=true          # set to false to reuse same endpoint name
SUFFIX_LEN="${SUFFIX_LEN:-5}"

# Compute & models
COMPUTE_NAME="batch-cluster"

TORCH_MODEL_NAME="mnist-classifier-torch"
TORCH_MODEL_PATH="deployment-torch/model"
TORCH_DEPLOYMENT_NAME="mnist-torch-dpl"
TORCH_DEPLOYMENT_FILE="deployment-torch/deployment.yml"

KERAS_MODEL_NAME="mnist-classifier-keras"
KERAS_MODEL_PATH="deployment-keras/model"
KERAS_DEPLOYMENT_NAME="mnist-keras-dpl"
KERAS_DEPLOYMENT_FILE="deployment-keras/deployment.yml"

# Endpoint spec
ENDPOINT_FILE="endpoint.yml"

# MNIST public input (reused everywhere)
MNIST_INPUT_URI="https://azuremlexampledata.blob.core.windows.net/data/mnist/sample"
MNIST_INPUT_TYPE="uri_folder"

# Download output target name for jobs (matches deployment YAML output name)
OUTPUT_PORT_NAME="score"

########################################
# Utilities
########################################

random_suffix() {
  tr -dc 'a-zA-Z0-9' </dev/urandom | fold -w "${SUFFIX_LEN}" | head -n 1
}

log() {
  # timestamped log line
  echo -e "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

exists_endpoint() {
  local name="$1"
  az ml batch-endpoint show --name "$name" >/dev/null 2>&1
}

exists_compute() {
  local name="$1"
  az ml compute show --name "$name" >/dev/null 2>&1
}

exists_model() {
  local name="$1"
  az ml model show --name "$name" >/dev/null 2>&1
}

exists_deployment() {
  local deployment_name="$1"
  local endpoint_name="$2"
  az ml batch-deployment show --name "$deployment_name" --endpoint-name "$endpoint_name" >/dev/null 2>&1
}

check_status_or_fail() {
  local job_name="$1"
  local status
  status="$(az ml job show -n "$job_name" --query status -o tsv)"
  log "Final job status: $status"
  if [[ "$status" == "Failed" ]]; then
    log "Job failed."
    exit 1
  fi
  if [[ "$status" != "Completed" ]]; then
    log "Job status is not 'Completed' (got '$status')."
    exit 2
  fi
}

invoke_and_wait() {
  # Usage:
  # invoke_and_wait <ENDPOINT_NAME> [additional az ml batch-endpoint invoke args...]
  local endpoint_name="$1"
  shift

  log "Invoking batch endpoint '$endpoint_name'..."
  local job_name
  # Note: using --query name -o tsv to capture job name from invoke
  job_name="$(az ml batch-endpoint invoke --name "$endpoint_name" "$@" --query name -o tsv)"
  log "Invoked job: $job_name"

  log "Opening job in Azure ML Studio..."
  az ml job show -n "$job_name" --web >/dev/null 2>&1 || true

  log "Streaming logs (follow) to console..."
  az ml job stream -n "$job_name"

  check_status_or_fail "$job_name"
  echo "$job_name"
}

########################################
# Ensure resources (idempotent)
########################################

ensure_compute() {
  local compute_name="$1"
  if exists_compute "$compute_name"; then
    log "Compute '$compute_name' exists. Skipping create."
  else
    log "Creating compute '$compute_name'..."
    az ml compute create \
      --name "$compute_name" \
      --type amlcompute \
      --min-instances 0 \
      --max-instances 5
  fi
}

ensure_model() {
  local model_name="$1"
  local model_path="$2"
  if exists_model "$model_name"; then
    log "Model '$model_name' exists. Skipping register."
  else
    log "Registering model '$model_name' (path: $model_path)..."
    az ml model create \
      --name "$model_name" \
      --type custom_model \
      --path "$model_path"
  fi
}

ensure_endpoint() {
  local endpoint_name="$1"
  local endpoint_file="$2"
  if exists_endpoint "$endpoint_name"; then
    log "Batch endpoint '$endpoint_name' exists. Showing details."
    az ml batch-endpoint show --name "$endpoint_name"
  else
    log "Creating batch endpoint '$endpoint_name'..."
    az ml batch-endpoint create --file "$endpoint_file" --name "$endpoint_name"
    log "Showing endpoint details..."
    az ml batch-endpoint show --name "$endpoint_name"
  fi
}

ensure_deployment() {
  # ensure_deployment <DEPLOYMENT_NAME> <ENDPOINT_NAME> <DEPLOYMENT_FILE> [set_default:true|false]
  local deployment_name="$1"
  local endpoint_name="$2"
  local deployment_file="$3"
  local set_default="${4:-false}"

  if exists_deployment "$deployment_name" "$endpoint_name"; then
    log "Deployment '$deployment_name' on endpoint '$endpoint_name' exists. Showing details."
    az ml batch-deployment show --name "$deployment_name" --endpoint-name "$endpoint_name"
  else
    log "Creating batch deployment '$deployment_name' on endpoint '$endpoint_name'..."
    if [[ "$set_default" == "true" ]]; then
      az ml batch-deployment create --file "$deployment_file" --endpoint-name "$endpoint_name" --set-default
    else
      az ml batch-deployment create --file "$deployment_file" --endpoint-name "$endpoint_name"
    fi
    log "Showing deployment details..."
    az ml batch-deployment show --name "$deployment_name" --endpoint-name "$endpoint_name"
  fi
}

########################################
# Resolve endpoint name
########################################

ENDPOINT_NAME="$ENDPOINT_BASE_NAME"
if [[ "$USE_SUFFIX" == "true" ]]; then
  ENDPOINT_NAME="${ENDPOINT_BASE_NAME}-$(random_suffix)"
fi
log "Using endpoint name: $ENDPOINT_NAME"

########################################
# Provisioning
########################################

# Models
ensure_model "$TORCH_MODEL_NAME" "$TORCH_MODEL_PATH"
ensure_model "$KERAS_MODEL_NAME" "$KERAS_MODEL_PATH"

# Compute
ensure_compute "$COMPUTE_NAME"

# Endpoint
ensure_endpoint "$ENDPOINT_NAME" "$ENDPOINT_FILE"

# Torch deployment (set as default initially)
ensure_deployment "$TORCH_DEPLOYMENT_NAME" "$ENDPOINT_NAME" "$TORCH_DEPLOYMENT_FILE" "true"

########################################
# Test default (Torch) deployment
########################################

log "Invoking endpoint with default deployment (Torch) using MNIST public URI..."
DEFAULT_JOB_NAME="$(invoke_and_wait "$ENDPOINT_NAME" \
  --input "$MNIST_INPUT_URI" \
  --input-type "$MNIST_INPUT_TYPE")"

log "Downloading scores to local path..."
az ml job download --name "$DEFAULT_JOB_NAME" --output-name "$OUTPUT_PORT_NAME" --download-path ./outputs-default

log "List jobs under deployment '$TORCH_DEPLOYMENT_NAME'..."
az ml batch-deployment list-jobs --name "$TORCH_DEPLOYMENT_NAME" --endpoint-name "$ENDPOINT_NAME" --query "[].name"

########################################
# Add non-default (Keras) deployment & test it
########################################

ensure_deployment "$KERAS_DEPLOYMENT_NAME" "$ENDPOINT_NAME" "$KERAS_DEPLOYMENT_FILE" "false"

log "Invoking non-default deployment '$KERAS_DEPLOYMENT_NAME'..."
KERAS_JOB_NAME="$(invoke_and_wait "$ENDPOINT_NAME" \
  --deployment-name "$KERAS_DEPLOYMENT_NAME" \
  --input "$MNIST_INPUT_URI" \
  --input-type "$MNIST_INPUT_TYPE")"

log "Downloading Keras job outputs..."
az ml job download --name "$KERAS_JOB_NAME" --output-name "$OUTPUT_PORT_NAME" --download-path ./outputs-keras

########################################
# Update endpoint default deployment to Keras
########################################

log "Updating default deployment to '$KERAS_DEPLOYMENT_NAME'..."
az ml batch-endpoint update --name "$ENDPOINT_NAME" --set defaults.deployment_name="$KERAS_DEPLOYMENT_NAME"

log "Verify default deployment:"
az ml batch-endpoint show --name "$ENDPOINT_NAME" --query "{Name: name, Defaults: defaults}"

########################################
# Invoke using new default + variants
########################################

log "Invoking endpoint using new default (Keras)..."
NEW_DEFAULT_JOB_NAME="$(invoke_and_wait "$ENDPOINT_NAME" \
  --input "$MNIST_INPUT_URI" \
  --input-type "$MNIST_INPUT_TYPE")"

log "Invoking with specific output file name..."
OUTPUT_FILE_NAME="predictions_${RANDOM}.csv"
OUTPUT_PATH="azureml:/datastores/workspaceblobstore/paths/${ENDPOINT_NAME}"
NAMED_OUTPUT_JOB_NAME="$(invoke_and_wait "$ENDPOINT_NAME" \
  --input "$MNIST_INPUT_URI" \
  --input-type "$MNIST_INPUT_TYPE" \
  --output-path "$OUTPUT_PATH" \
  --set output_file_name="$OUTPUT_FILE_NAME")"

log "Invoking with overwrite parameters (mini-batch size & instance count)..."
OVERWRITE_JOB_NAME="$(invoke_and_wait "$ENDPOINT_NAME" \
  --input "$MNIST_INPUT_URI" \
  --input-type "$MNIST_INPUT_TYPE" \
  --mini-batch-size 20 \
  --instance-count 5)"

########################################
# Optional cleanup (uncomment to use)
########################################
# log "Deleting deployment '$TORCH_DEPLOYMENT_NAME'..."
# az ml batch-deployment delete --name "$TORCH_DEPLOYMENT_NAME" --endpoint-name "$ENDPOINT_NAME" --yes
#
# log "Deleting endpoint '$ENDPOINT_NAME'..."
# az ml batch-endpoint delete --name "$ENDPOINT_NAME" --yes

log "Done."
