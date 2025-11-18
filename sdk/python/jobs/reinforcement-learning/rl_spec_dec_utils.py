"""
Utility functions for RL training and Speculative Decoding deployment.
This module abstracts common operations for the notebook.
"""

import os
import time
import uuid
import json
import shutil
import requests
from pathlib import Path
from huggingface_hub import snapshot_download
from azure.ai.ml import MLClient, Input, dsl
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import (
    Model,
    KubernetesOnlineEndpoint,
    KubernetesOnlineDeployment,
    ProbeSettings,
    OnlineRequestSettings,
)

ml_client = None
registry_ml_client = None

class RLSpecDecPipeline:
    """Main class for managing RL training and Speculative Decoding workflow."""

    def __init__(self):
        # We use an unique identifier for naming resources, this prevents name collisions for resources created in this lab
        self.guid = str(uuid.uuid4())[:8]

    def create_rl_pipeline(
        self,
        huggingface_id,
        train_data_asset,
        val_data_asset,
        compute_cluster,
        config={},
    ):
        """Create and submit RL pipeline job using registry component."""
        print("Creating RL pipeline...")

        # Use defaults to ensure reproducibility and avoid missing params
        default_config = {
            "experiment_name": "reinforcement-learning-grpo",
            "instance_type_finetune": "octagpu",
            "instance_type_model_import": "octacpu",
            "num_nodes_finetune": 2,
            "number_of_gpu_to_use_finetuning": 8,
            "algorithm_adv_estimator": "grpo",
            "data_max_prompt_length": 8192,
            "actor_strategy": "fsdp",
            "trainer_total_epochs": 1,
            "actor_fsdp_config_mixed_precision_reduce_dtype": "bf16",
            "actor_fsdp_config_mixed_precision_buffer_dtype": "bf16",
        }
        default_config.update(config)  # Allow user override for flexibility

        # Extract experiment_name from config as that is passed separately
        if "experiment_name" in default_config:
            experiment_name = default_config["experiment_name"]
            del default_config["experiment_name"]

        # Use registry component for versioning and reuse
        pipeline_component_func = registry_ml_client.components.get(
            name="pipeline_rl_finetune",
            label="latest"
        )

        # Define pipeline to encapsulate all steps for traceability and reuse
        @dsl.pipeline(name=f"grpo-finqa-{self.guid}")
        def create_pipeline():
            rl_pipeline = pipeline_component_func(
                huggingface_id=huggingface_id,
                compute_model_import=compute_cluster,
                compute_finetune=compute_cluster,
                data_train_files=Input(type=AssetTypes.URI_FILE, path=train_data_asset.id),
                data_val_files=Input(type=AssetTypes.URI_FILE, path=val_data_asset.id),
                **default_config, # Pass all config as kwargs for maintainability and future-proofing
            )
            return {"model_output": rl_pipeline.outputs.model_output}

        pipeline_object = create_pipeline()

        # Force rerun to ensure new job, avoid stale results
        pipeline_object.settings.force_rerun = True
        pipeline_object.settings.continue_on_step_failure = False  # Fail fast for debugging

        # Submit job
        print("Submitting pipeline...")
        rl_run = ml_client.jobs.create_or_update(
            pipeline_object,
            experiment_name=experiment_name,
        )
        print(f"Studio URL: {rl_run.studio_url}")  # Clickable link for monitoring

        return rl_run

    def monitor_job(self, job_name):
        print(f"Monitoring job: {job_name}")
        print(f"Checking every 30 seconds...")

        # Polling loop: avoids AzureML SDK timeouts and gives live status
        while True:
            job = ml_client.jobs.get(job_name)
            status = job.status
            
            print(f"[{time.strftime('%H:%M:%S')}] Status: {status}")

            # Exit on terminal state to avoid infinite loop and allow downstream logic
            if status in ["Completed", "Failed", "Canceled"]:
                return job, status
            time.sleep(30)  # Throttle polling to avoid API rate limits

    def register_model(self, job, model_name_prefix):
        print("Registering model from job output...")

        # Use GUID to ensure model name uniqueness across runs
        model_name = f"{model_name_prefix}-{self.guid}"
        model_output = job.outputs.model_output

        # Assets must be registered as models before use in endpoints 
        model = Model(
            path=model_output.path,
            name=model_name,
            description="GRPO fine-tuned model on FinQA",
            type=AssetTypes.CUSTOM_MODEL,
        )

        registered_model = ml_client.models.create_or_update(model)  # Register the model
        print(f"Model: {registered_model.name} v{registered_model.version}")  
        print(f"ID: {registered_model.id}")

        return registered_model

    def create_spec_dec_endpoint(
        self,
        base_model,
        instance_type="monogpu", # Kubernetes supports partial node usage granular upto the GPU level
        compute_name="shj-a100",
    ):
        """Create speculative decoding endpoint using Kubernetes."""
        # Use a fixed mount path for model assets to standardize deployment layout
        model_mount_path = "/var/model-mount"
        print("🌐 Creating speculative decoding endpoint...")

        # Unique names prevent collisions and allow parallel experiments
        endpoint_name = f"spec-dec-{self.guid}"
        deployment_name = "spec-dec-deploy"

        # Use AzureML endpoint abstraction for traffic management and auth
        endpoint = KubernetesOnlineEndpoint(
            name=endpoint_name,
            auth_mode="key",
            compute=compute_name,
        )

        print(f"  ✓ Creating endpoint: {endpoint_name}")
        ml_client.online_endpoints.begin_create_or_update(endpoint).wait()

        # Probes are APIs exposed by the deployment which informs the framework if the deployment is healthy and ready to receive traffic
        probe_settings = ProbeSettings(
            initial_delay=1400,
            period=30,
            timeout=2,
            success_threshold=1,
            failure_threshold=30,
        )

        # SGLANG environment variables
        environment_variables = {
            "SPECULATIVE_DECODING_MODE": "true",
            "BASE_MODEL": f"{model_mount_path}/models/base",
            "DRAFT_MODEL": f"{model_mount_path}/models/draft",
            "NUM_SPECULATIVE_TOKENS": "5",
            "SERVING_ENGINE": "sglang",
        }

        # Use deployment abstraction for scaling, versioning, and isolation
        deployment = KubernetesOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=base_model.id,
            model_mount_path=model_mount_path,
            instance_type=instance_type,
            instance_count=1,
            environment=ml_client.environments.get("speculative-decoding-env", label="latest"),
            environment_variables=environment_variables,
            liveness_probe=probe_settings,
            readiness_probe=probe_settings,
            request_settings=OnlineRequestSettings(
                request_timeout_ms=90000,
                max_concurrent_requests_per_instance=4,
            ),
        )

        print(f"  ✓ Creating deployment (15-20 min)...")
        ml_client.online_deployments.begin_create_or_update(deployment).wait()

        # Route all traffic to new deployment for immediate use
        endpoint.traffic = {deployment_name: 100}
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()

        print(f"  ✓ Endpoint ready: {endpoint_name}")

        return endpoint_name

    def test_endpoint(self, endpoint_name):
        """Test endpoint with a sample request.
        Validates that the deployed endpoint is live and returns expected results, ensuring end-to-end functionality.
        """
        print("🧪 Testing endpoint...")
        # Retrieve endpoint URI and API key to authenticate test request
        scoring_uri = ml_client.online_endpoints.get(endpoint_name).scoring_uri
        api_key = ml_client.online_endpoints.get_keys(endpoint_name).primary_key

        # Use a realistic financial question to verify model reasoning and output format
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": """Please answer the following financial question:

Context: A company has revenue of $1,000,000 and expenses of $750,000.

Question: What is the profit margin as a percentage?
Let's think step by step and put final answer after ####."""
                }
            ],
            "max_tokens": 512,
            "temperature": 0.7,
        }

        # Set headers for JSON content and bearer authentication
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        response = requests.post(scoring_uri, json=payload, headers=headers)

        if response.status_code == 200:
            result = response.json()
            # Extract the model response
            if "choices" in result and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"]
                print(f"  ✓ Response received")
                print(f"\n{'='*60}")
                print(answer)
                print(f"{'='*60}\n")
                return result
        else:
            print(f"  ✗ Error: {response.status_code}")
            print(f"  {response.text}")
            return None


def create_draft_model_config(base_model_config=None):
    """Combines user config and draft model configuration for EAGLE3."""
    default_config = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "max_position_embeddings": 2048,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 1,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-05,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "4.28.1",
        "use_cache": True,
        "vocab_size": 128256,
        "draft_vocab_size": 32000
    }

    if base_model_config:
        default_config.update(base_model_config)

    return default_config


def setup_workspace(config_path="./config.json", registry_name="test_centralus"):
    """Setup Azure ML workspace and registry clients."""
    global ml_client, registry_ml_client
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception:
        credential = InteractiveBrowserCredential()

    ml_client = MLClient.from_config(credential=credential, path=config_path)
    _ = ml_client._workspaces.get(ml_client.workspace_name) # Load credentials to verify access
    registry_ml_client = MLClient(credential, registry_name=registry_name)

    print(f"Workspace setup complete, connected")
    return ml_client, registry_ml_client


def run_rl_training_pipeline(
    base_model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    compute_cluster="h100-dedicated",
    config={},
):
    print(" Starting RL Training Pipeline")
    pipeline = RLSpecDecPipeline()

    # We have uploaded the data assets to our registry in advance for this tutorial
    train_asset = ml_client.data.get(name="dataset_training_finqa", label="latest")
    val_asset = ml_client.data.get(name="dataset_validation_finqa", label="latest")

    # Submit RL pipeline job with all required config and assets
    rl_job = pipeline.create_rl_pipeline(
        huggingface_id=base_model_id,
        train_data_asset=train_asset,
        val_data_asset=val_asset,
        compute_cluster=compute_cluster,
        config=config,
    )
    
    completed_job, status = pipeline.monitor_job(rl_job.name)
    if status == "Completed":
        # Register the trained model for downstream deployment and tracking
        registered_model = pipeline.register_model(
            job=completed_job,
            model_name_prefix="grpo-finqa-model",
            base_model_id=base_model_id,
        )
        return rl_job, status, registered_model
    else:
        print(f"\n Job did not complete successfully: {status}")
        return rl_job, status, None


def download_and_register_hf_model(hf_model_id, azureml_model_name): #[WIP]
    guid = str(uuid.uuid4())[:4]
    temp_dir = f"./models/temp-{guid}/model_artifact/model"
    os.makedirs(temp_dir, exist_ok=True)
    snapshot_download(repo_id=hf_model_id, local_dir=temp_dir)

    model_folder = Model(
        path=temp_dir,
        type=AssetTypes.MLFLOW_MODEL,
        name=azureml_model_name,
    )
    model = ml_client.models.create_or_update(model_folder)
    return model


def run_draft_model_pipeline(
    compute_cluster="h100-dedicated",
    base_model_mlflow_path="azureml://registries/azureml-meta/models/Meta-Llama-3-8B-Instruct/versions/9",
    draft_train_data_path="./data_for_draft_model/train/sharegpt_train_small.jsonl",
    num_epochs=1,
    monitor=False,
):
    print("\n" + "="*60)
    print("🎯 STARTING DRAFT MODEL PIPELINE")
    print("="*60 + "\n")

    # Create draft model config
    draft_model_config = create_draft_model_config()

    config_dir = "./draft_config"
    os.makedirs(config_dir, exist_ok=True)
    draft_config_path = os.path.join(config_dir, "draft_model_config.json")

    with open(draft_config_path, "w") as f:
        json.dump(draft_model_config, f, indent=4)
    print(f"  ✓ Draft model config saved: {draft_config_path}")

    # Verify training data
    if not os.path.exists(draft_train_data_path):
        raise FileNotFoundError(f"Draft model training data not found: {draft_train_data_path}")
    print(f"  ✓ Draft training data: {draft_train_data_path}")

    # Load component
    draft_component_name = "eagle3_chat_completion_pipeline"
    print(f"  ✓ Loading component: {draft_component_name}")
    eagle3_comp = registry_ml_client.components.get(name=draft_component_name, label="latest")
    print(f"  ✓ Component loaded: {eagle3_comp.name} v{eagle3_comp.version}")

    # Define pipeline
    @pipeline
    def speculative_decoding_draft_pipeline():
        node = eagle3_comp(
            mlflow_model_path=Input(type=AssetTypes.MLFLOW_MODEL, path=base_model_mlflow_path),
            dataset_train_split=Input(type=AssetTypes.URI_FILE, path=draft_train_data_path),
            dataset_validation_split=Input(type=AssetTypes.URI_FILE, path=draft_train_data_path),
            draft_model_config=Input(type=AssetTypes.URI_FILE, path=draft_config_path),
            compute_model_import=compute_cluster,
            compute_eagle3_training=compute_cluster,
            instance_type_model_import="octacpu",
            instance_type_eagle3_training="octagpu",
            num_epochs=num_epochs,
        )
        return {"output_model": node.outputs.output_model_path}

    # Submit pipeline
    draft_job = speculative_decoding_draft_pipeline()
    print("  ✓ Submitting draft model training pipeline...")
    draft_job = ml_client.jobs.create_or_update(
        draft_job, experiment_name="speculative-decoding-draft-model"
    )

    print(f"  ✓ Job submitted: {draft_job.name}")
    print(f"  📊 Studio URL: {draft_job.studio_url}")

    # Monitor if requested
    if monitor:
        pipeline = RLSpecDecPipeline()
        _, status = pipeline.monitor_job(draft_job.name, poll_interval=60)
        return draft_job, status

    return draft_job, None


def prepare_combined_model_for_deployment(
    draft_job_name,
    base_model_hf_id="nvidia/Llama-3.1-8B-Instruct-FP8",
    model_name="grpo-speculative-decoding",
    force=False,
):
    print("\n" + "="*60)
    print("📦 PREPARING COMBINED MODEL FOR DEPLOYMENT")
    print("="*60 + "\n")

    draft_pipeline = DraftModelPipeline()

    # Define paths
    draft_model_dir = "./models/draft"
    base_model_dir = "./models/base"

    temp_download_dir = "./models/draft_temp"
    temp_path = Path(temp_download_dir)
    required_files = ["config.json", "model.safetensors", "training_state.pt"]

    for file_pattern in required_files:
        files_found = list(temp_path.rglob(file_pattern))
        if files_found:
            src_path = files_found[0]  # Take the first match
            dst_path = Path(draft_model_dir) / file_pattern
            shutil.move(str(src_path), str(dst_path))
            print(f"  ✓ Moved {file_pattern}")
        else:
            print(f"  ⚠️ File not found: {file_pattern}")

    # Clean up temporary directory
    if os.path.exists(temp_download_dir):
        shutil.rmtree(temp_download_dir)
        print(f"  ✓ Cleaned up temporary directory")
    else:
        print(f"  ✓ Draft model already exists: {draft_model_dir}")

    # Download base model from HuggingFace
    if force or not os.path.exists(base_model_dir):
        print("\n📥 Downloading base model...")
        snapshot_download(repo_id=base_model_hf_id, local_dir=base_model_dir)
        print(f"  ✓ Base model downloaded to: {base_model_dir}")
    else:
        print(f"  ✓ Base model already exists: {base_model_dir}")

    # Upload combined model
    combined_model = draft_pipeline.upload_combined_model(
        base_model_dir=base_model_dir,
        draft_model_dir=draft_model_dir,
        model_name=model_name,
    )

    print(f"\n✓ Combined model ready for deployment: {combined_model.name}")
    return combined_model


def deploy_speculative_decoding_endpoint(
    combined_model,
    instance_type="monogpu",
    compute_name="shj-a100",
):
    print("Deploying speculative decoding endpoint")

    pipeline = RLSpecDecPipeline()
    endpoint_name = f"spec-dec-grpo-{pipeline.guid}"
    deployment_name = "speculative-deployment"
    model_mount_path = "/var/model-mount"

    print(f"Creating endpoint: {endpoint_name}")
    endpoint = KubernetesOnlineEndpoint(
        name=endpoint_name,
        description="Speculative decoding endpoint with GRPO fine-tuned base model",
        auth_mode="key",
        compute=compute_name,
        tags={"model_type": "speculative_decoding", "algorithm": "grpo"},
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).wait()

    # Configure probes
    probe_settings = ProbeSettings(
        initial_delay=1400,
        period=30,
        timeout=2,
        success_threshold=1,
        failure_threshold=30,
    )

    # Environment variables
    environment_variables = {
        "SPECULATIVE_DECODING_MODE": "true",
        "BASE_MODEL": f"{model_mount_path}/models/base",
        "DRAFT_MODEL": f"{model_mount_path}/models/draft",
    }

    deployment = KubernetesOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=combined_model.id,
        model_mount_path=model_mount_path,
        instance_type=instance_type,
        instance_count=1,
        environment=ml_client.environments.get("speculative-decoding-env", label="latest"), 
        environment_variables=environment_variables,
        liveness_probe=probe_settings,
        readiness_probe=probe_settings,
        request_settings=OnlineRequestSettings(
            request_timeout_ms=90000,
            max_concurrent_requests_per_instance=4,
        ),
    )

    print(f"  ✓ Creating deployment (this takes 15-20 min)...")
    ml_client.online_deployments.begin_create_or_update(deployment).wait()

    # Route traffic
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()

    print(f"✓ Speculative decoding endpoint deployed: {endpoint_name}")
    return endpoint_name


def test_deployment(endpoint_name):
    print("Testing deployment")

    pipeline = RLSpecDecPipeline()
    endpoint_info = pipeline.get_endpoint_details(endpoint_name)

    print(f"Endpoint: {endpoint_info['endpoint_name']}")

    result = pipeline.test_endpoint(
        scoring_uri=endpoint_info['scoring_uri'],
        api_key=endpoint_info['api_key'],
    )

    return result


class DraftModelPipeline:
    """Class for managing draft model creation for speculative decoding."""

    def __init__(self):
        self.guid = str(uuid.uuid4())[:8]

    def create_draft_model_pipeline(
        self,
        base_model_path,
        training_data_path,
        validation_data_path=None,
        draft_model_config=None,
        compute_name=None,
        num_epochs=1,
        component_name="speculative_decoding_draft_pipeline",
    ):
        print("Creating draft model pipeline...")

        # Use validation data same as training if not provided
        if validation_data_path is None:
            validation_data_path = training_data_path

        # Get draft model configuration
        if draft_model_config is None:
            draft_model_config = create_draft_model_config()

        # Save draft config locally
        config_dir = "./draft_config"
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, f"draft_config_{self.guid}.json")

        with open(config_path, "w") as f:
            json.dump(draft_model_config, f, indent=4)

        try:
            pipeline_component_func = registry_ml_client.components.get(
                name=component_name,
                label="latest"
            )
            print(f"  ✓ Component loaded: {pipeline_component_func.name} v{pipeline_component_func.version}")
        except Exception as e:
            print(f"  ✗ Failed to load component: {e}")
            print(f"  ℹ️  Make sure component '{component_name}' exists in registry")
            raise

        # Define the pipeline job
        @dsl.pipeline(name=f"draft-model-{self.guid}")
        def create_pipeline():
            draft_pipeline = pipeline_component_func(
                mlflow_model_path=Input(type=AssetTypes.MLFLOW_MODEL, path=base_model_path),
                dataset_train_split=Input(type=AssetTypes.URI_FILE, path=training_data_path),
                dataset_validation_split=Input(type=AssetTypes.URI_FILE, path=validation_data_path),
                draft_model_config=Input(type=AssetTypes.URI_FILE, path=config_path),
                compute_model_import=compute_name,
                compute_eagle3_training=compute_name,
                num_epochs=num_epochs,
            )
            return {"output_model": draft_pipeline.outputs.output_model_path}

        # Create pipeline object
        pipeline_object = create_pipeline()

        # Don't use cached results
        pipeline_object.settings.force_rerun = True
        pipeline_object.settings.continue_on_step_failure = False

        # Submit job
        print("  ✓ Submitting draft model pipeline...")
        draft_run = ml_client.jobs.create_or_update(
            pipeline_object,
            experiment_name="speculative-decoding-draft-model"
        )

        print(f"  ✓ Job submitted: {draft_run.name}")
        print(f"  📊 Studio URL: {draft_run.studio_url}")

        return draft_run

    def download_draft_model(self, job_name, output_dir="./models/draft"):
        """Download draft model from completed pipeline job."""
        print(f"📥 Downloading draft model from job: {job_name}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Download model output
        ml_client.jobs.download(
            name=job_name,
            output_name="output_model",
            download_path=output_dir,
            all=True
        )

        print(f"  ✓ Draft model downloaded to: {output_dir}")

        # Flatten directory structure
        self._flatten_directory(output_dir)

        # Update config with extended context
        self._update_draft_config(output_dir)

        return output_dir

    def _flatten_directory(self, directory):
        """Move all files from subdirectories to root."""

        print("  ✓ Flattening directory structure...")

        for root, dirs, files in os.walk(directory):
            for file in files:
                if root != directory:
                    source = os.path.join(root, file)
                    destination = os.path.join(directory, file)
                    if not os.path.exists(destination):
                        shutil.move(source, destination)

        # Remove empty subdirectories
        for root, dirs, files in os.walk(directory, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)

    def _update_draft_config(self, model_dir):
        """Update draft model config with extended context settings."""

        config_path = os.path.join(model_dir, "config.json")

        if not os.path.exists(config_path):
            print("  ⚠️  config.json not found, skipping update")
            return

        print("  ✓ Updating draft model config...")

        with open(config_path, "r") as f:
            draft_config = json.load(f)

        # Update with extended context settings
        draft_config.update({
            "max_position_embeddings": 131072,
            "rope_scaling": {
                "factor": 8,
                "high_freq_factor": 4,
                "low_freq_factor": 1,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            }
        })

        with open(config_path, "w") as f:
            json.dump(draft_config, f, indent=4)

        print("  ✓ Config updated with extended context settings")

    def upload_combined_model(
        self,
        base_model_dir,
        draft_model_dir,
        model_name="speculative-decoding-combined",
    ):
        """Upload base and draft models as a combined custom model."""

        print("📦 Creating combined model package...")
        combined_dir = "./models/"
        print(f"  ✓ Base model: {base_model_dir}")
        print(f"  ✓ Draft model: {draft_model_dir}")

        # Register combined model
        model_name_versioned = f"{model_name}-{self.guid}"

        model = Model(
            path=combined_dir,
            name=model_name_versioned,
            description="Combined base and draft model for speculative decoding",
            tags={
                "type": "speculative_decoding",
                "architecture": "eagle3",
            }
        )

        registered_model = ml_client.models.create_or_update(model)
        print(f"  ✓ Model registered: {registered_model.name} v{registered_model.version}")

        return registered_model
