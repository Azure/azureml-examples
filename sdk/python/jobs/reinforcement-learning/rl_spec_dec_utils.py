"""
Utility functions for RL training and Speculative Decoding deployment.
This module abstracts common operations for the notebook.
"""

import os
import time
import uuid
import json
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from azure.ai.ml import MLClient, Input, Output, dsl
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import (
    Data,
    Model,
    KubernetesOnlineEndpoint,
    KubernetesOnlineDeployment,
    ProbeSettings,
    OnlineRequestSettings,
)


# Dataset paths
TRAIN_DATASET_PATH = os.path.join(os.getcwd(), "datasets", "train_finqa.jsonl")
VALIDATION_DATASET_PATH = os.path.join(os.getcwd(), "datasets", "validation_finqa.jsonl")


class RLSpecDecPipeline:
    """Main class for managing RL training and Speculative Decoding workflow."""

    def __init__(self, workspace_ml_client):
        """Initialize with Azure ML workspace client."""
        self.ml_client = workspace_ml_client
        self.guid = str(uuid.uuid4())[:8]

    def register_datasets(self, train_path, val_path):
        """Register training and validation datasets."""
        print("📁 Registering datasets...")

        # Register training dataset
        train_dataset_name = f"finqa_train_{self.guid}"
        train_data = Data(
            path=train_path,
            type=AssetTypes.URI_FILE,
            description="FinQA training dataset for RL",
            name=train_dataset_name,
            version="1",
        )
        train_asset = self.ml_client.data.create_or_update(train_data)
        print(f"  ✓ Training dataset: {train_asset.name}")

        # Register validation dataset
        val_dataset_name = f"finqa_validation_{self.guid}"
        val_data = Data(
            path=val_path,
            type=AssetTypes.URI_FILE,
            description="FinQA validation dataset for RL",
            name=val_dataset_name,
            version="1",
        )
        val_asset = self.ml_client.data.create_or_update(val_data)
        print(f"  ✓ Validation dataset: {val_asset.name}")

        return train_asset, val_asset

    def create_rl_pipeline(
        self,
        registry_ml_client,
        huggingface_id,
        train_data_asset,
        val_data_asset,
        compute_cluster,
        config=None,
        pipeline_component_name="arl_finetune_pipeline",
    ):
        """Create and submit RL pipeline job using registry component."""
        print("🚀 Creating RL pipeline...")

        # Default configuration
        default_config = {
            "experiment_name": "reinforcement-learning-grpo",
            "instance_type_finetune": "octagpu",
            "instance_type_model_import": "octacpu",
            "num_nodes_finetune": 1,
            "number_of_gpu_to_use_finetuning": 8,
            "algorithm_adv_estimator": "grpo",
            "trainer_total_epochs": 15,
            "actor_optim_lr": 3e-6,
            "data_train_batch_size": 512,
            "data_max_prompt_length": 1024,
            "data_max_response_length": 2048,
            "actor_model_lora_rank": 64,
            "actor_model_lora_alpha": 32,
            "actor_strategy": "fsdp2",
        }

        if config:
            default_config.update(config)

        # Fetch the pipeline component from registry
        print(f"  ✓ Loading pipeline component: {pipeline_component_name}")
        try:
            pipeline_component_func = registry_ml_client.components.get(
                name=pipeline_component_name,
                label="latest"
            )
            print(f"  ✓ Component loaded: {pipeline_component_func.name} v{pipeline_component_func.version}")
        except Exception as e:
            print(f"  ✗ Failed to load component: {e}")
            print(f"  ℹ️  Make sure component '{pipeline_component_name}' exists in registry")
            raise

        # Define the pipeline job
        @dsl.pipeline(name=f"grpo-finqa-{self.guid}")
        def create_pipeline():
            rl_pipeline = pipeline_component_func(
                huggingface_id=huggingface_id,
                compute_model_import=compute_cluster,
                compute_finetune=compute_cluster,
                data_train_files=Input(type=AssetTypes.URI_FILE, path=train_data_asset.id),
                data_val_files=Input(type=AssetTypes.URI_FILE, path=val_data_asset.id),
                instance_type_model_import=default_config["instance_type_model_import"],
                instance_type_finetune=default_config["instance_type_finetune"],
                num_nodes_finetune=default_config["num_nodes_finetune"],
                number_of_gpu_to_use_finetuning=default_config["number_of_gpu_to_use_finetuning"],
                algorithm_adv_estimator=default_config["algorithm_adv_estimator"],
                trainer_total_epochs=default_config["trainer_total_epochs"],
                actor_optim_lr=default_config["actor_optim_lr"],
                data_train_batch_size=default_config["data_train_batch_size"],
                data_max_prompt_length=default_config["data_max_prompt_length"],
                data_max_response_length=default_config["data_max_response_length"],
                actor_model_lora_rank=default_config["actor_model_lora_rank"],
                actor_model_lora_alpha=default_config["actor_model_lora_alpha"],
                actor_strategy=default_config["actor_strategy"],
            )
            return {"model_output": rl_pipeline.outputs.model_output}

        # Create pipeline object
        pipeline_object = create_pipeline()

        # Don't use cached results from previous jobs
        pipeline_object.settings.force_rerun = True
        pipeline_object.settings.continue_on_step_failure = False

        # Submit job
        print("  ✓ Submitting pipeline...")
        rl_run = self.ml_client.jobs.create_or_update(
            pipeline_object,
            experiment_name=default_config["experiment_name"]
        )
        print(f"  ✓ Job submitted: {rl_run.name}")
        print(f"  📊 Studio URL: {rl_run.studio_url}")

        return rl_run

    def monitor_job(self, job_name, poll_interval=60):
        """Monitor job status until completion."""
        print(f"⏳ Monitoring job: {job_name}")
        print(f"   Checking every {poll_interval} seconds...")

        while True:
            job = self.ml_client.jobs.get(job_name)
            status = job.status

            print(f"   [{time.strftime('%H:%M:%S')}] Status: {status}")

            if status in ["Completed", "Failed", "Canceled"]:
                print(f"\n{'✓' if status == 'Completed' else '✗'} Job {status}")
                return job, status

            time.sleep(poll_interval)

    def register_model(self, job, model_name_prefix, base_model_id):
        """Register fine-tuned model from job output."""
        print("📦 Registering model...")

        model_name = f"{model_name_prefix}-{self.guid}"
        model_output = job.outputs.model_output

        model = Model(
            path=model_output.path,
            name=model_name,
            description="GRPO fine-tuned model on FinQA",
            type=AssetTypes.CUSTOM_MODEL,
            tags={
                "algorithm": "grpo",
                "dataset": "finqa",
                "base_model": base_model_id,
            },
        )

        registered_model = self.ml_client.models.create_or_update(model)
        print(f"  ✓ Model: {registered_model.name} v{registered_model.version}")
        print(f"  📍 ID: {registered_model.id}")

        return registered_model

    def create_spec_dec_endpoint(
        self,
        base_model,
        instance_type="monogpu",
        compute_name="shj-a100",
    ):
        """Create speculative decoding endpoint using Kubernetes."""
        model_mount_path="/var/model-mount"
        print("🌐 Creating speculative decoding endpoint...")

        endpoint_name = f"spec-dec-{self.guid}"
        deployment_name = "spec-dec-deploy"

        # Create Kubernetes endpoint
        endpoint = KubernetesOnlineEndpoint(
            name=endpoint_name,
            description="Speculative Decoding with GRPO model",
            auth_mode="key",
            compute=compute_name,
            tags={"model_type": "speculative_decoding", "algorithm": "grpo"},
        )

        print(f"  ✓ Creating endpoint: {endpoint_name}")
        self.ml_client.online_endpoints.begin_create_or_update(endpoint).wait()

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
            "NUM_SPECULATIVE_TOKENS": "5",
            "SERVING_ENGINE": "sglang",
        }

        # Create Kubernetes deployment
        deployment = KubernetesOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=base_model.id,
            model_mount_path=model_mount_path,
            instance_type=instance_type,
            instance_count=1,
            environment=self.ml_client.environments.get("speculative-decoding-env", label="latest"), 
            environment_variables=environment_variables,
            liveness_probe=probe_settings,
            readiness_probe=probe_settings,
            request_settings=OnlineRequestSettings(
                request_timeout_ms=90000,
                max_concurrent_requests_per_instance=4,
            ),
        )

        print(f"  ✓ Creating deployment (15-20 min)...")
        self.ml_client.online_deployments.begin_create_or_update(deployment).wait()

        # Route traffic
        endpoint.traffic = {deployment_name: 100}
        self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()

        print(f"  ✓ Endpoint ready: {endpoint_name}")

        return endpoint_name

    def get_endpoint_details(self, endpoint_name):
        """Get endpoint URI and API key."""
        endpoint = self.ml_client.online_endpoints.get(endpoint_name)
        api_key = self.ml_client.online_endpoints.get_keys(endpoint_name).primary_key

        return {
            "scoring_uri": endpoint.scoring_uri,
            "api_key": api_key,
            "endpoint_name": endpoint_name,
        }

    def test_endpoint(self, scoring_uri, api_key):
        """Test endpoint with a sample request."""
        import requests
        import json

        print("🧪 Testing endpoint...")

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

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        response = requests.post(scoring_uri, json=payload, headers=headers)

        if response.status_code == 200:
            result = response.json()
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


def verify_datasets():
    """Verify dataset files exist and return their paths."""
    paths = {
        "train": TRAIN_DATASET_PATH,
        "validation": VALIDATION_DATASET_PATH,
    }

    print("🔍 Verifying datasets...")
    for name, path in paths.items():
        exists = os.path.exists(path)
        symbol = "✓" if exists else "✗"
        print(f"  {symbol} {name}: {path}")

        if not exists:
            raise FileNotFoundError(f"Dataset not found: {path}")

    return paths


def create_draft_model_config(base_model_config=None):
    """Create draft model configuration for EAGLE3."""
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
    from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception:
        credential = InteractiveBrowserCredential()

    ml_client = MLClient.from_config(credential=credential, path=config_path)
    workspace = ml_client._workspaces.get(ml_client.workspace_name)
    registry_ml_client = MLClient(credential, registry_name=registry_name)

    print(f"✓ Connected to registry: {registry_name}")
    print(f"✓ Connected to workspace: {workspace.name}")
    print(f"✓ Resource group: {ml_client.resource_group_name}")

    return ml_client, registry_ml_client


def run_rl_training_pipeline(
    ml_client,
    registry_ml_client,
    base_model_id="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    compute_cluster="h100-dedicated",
    training_config=None,
    monitor=True,
):
    """Run complete RL training pipeline from datasets to trained model."""
    print("\n" + "="*60)
    print(" Starting RL Training Pipeline")
    print("="*60 + "\n")

    # Initialize pipeline
    pipeline = RLSpecDecPipeline(ml_client)

    # Verify and register datasets
    dataset_paths = verify_datasets()
    train_asset, val_asset = pipeline.register_datasets(
        train_path=dataset_paths["train"],
        val_path=dataset_paths["validation"],
    )

    # Create and submit RL pipeline
    rl_job = pipeline.create_rl_pipeline(
        registry_ml_client=registry_ml_client,
        huggingface_id=base_model_id,
        train_data_asset=train_asset,
        val_data_asset=val_asset,
        compute_cluster=compute_cluster,
        config=training_config or {},
    )

    # Monitor if requested
    if monitor:
        completed_job, status = pipeline.monitor_job(rl_job.name, poll_interval=60)
        if status == "Completed":
            registered_model = pipeline.register_model(
                job=completed_job,
                model_name_prefix="grpo-finqa-model",
                base_model_id=base_model_id,
            )
            return rl_job, status, registered_model
        else:
            print(f"\n Job did not complete successfully: {status}")
            return rl_job, status, None

    return rl_job, None, None


def run_draft_model_pipeline(
    ml_client,
    registry_ml_client,
    compute_cluster="h100-dedicated",
    base_model_mlflow_path="azureml://registries/azureml-meta/models/Meta-Llama-3-8B-Instruct/versions/9",
    draft_train_data_path="./data_for_draft_model/train/sharegpt_train_small.jsonl",
    num_epochs=1,
    monitor=False,
):
    """Run complete draft model training pipeline."""


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
        pipeline = RLSpecDecPipeline(ml_client)
        _, status = pipeline.monitor_job(draft_job.name, poll_interval=60)
        return draft_job, status

    return draft_job, None


def prepare_combined_model_for_deployment(
    ml_client,
    draft_job_name,
    base_model_hf_id="nvidia/Llama-3.1-8B-Instruct-FP8",
    model_name="grpo-speculative-decoding",
    force=False,
):
    """Download, combine and register base+draft models for deployment."""
    print("\n" + "="*60)
    print("📦 PREPARING COMBINED MODEL FOR DEPLOYMENT")
    print("="*60 + "\n")

    draft_pipeline = DraftModelPipeline(ml_client)

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
    ml_client,
    combined_model,
    instance_type="monogpu",
    compute_name="shj-a100",
):
    """Deploy speculative decoding endpoint with combined model using Kubernetes."""
    print("\n" + "="*60)
    print("🌐 DEPLOYING SPECULATIVE DECODING ENDPOINT")
    print("="*60 + "\n")

    pipeline = RLSpecDecPipeline(ml_client)
    endpoint_name = f"spec-dec-grpo-{pipeline.guid}"
    deployment_name = "speculative-deployment"
    model_mount_path = "/var/model-mount"

    # Create Kubernetes endpoint
    print(f"  ✓ Creating endpoint: {endpoint_name}")
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

    # Create Kubernetes deployment
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


def test_deployment(ml_client, endpoint_name):
    """Test the deployed speculative decoding endpoint."""
    print("\n" + "="*60)
    print("🧪 TESTING DEPLOYMENT")
    print("="*60 + "\n")

    pipeline = RLSpecDecPipeline(ml_client)
    endpoint_info = pipeline.get_endpoint_details(endpoint_name)

    print(f"📍 Endpoint: {endpoint_info['endpoint_name']}")
    print(f"🔗 URI: {endpoint_info['scoring_uri']}")
    print(f"🔑 Key: {endpoint_info['api_key'][:10]}...\n")

    result = pipeline.test_endpoint(
        scoring_uri=endpoint_info['scoring_uri'],
        api_key=endpoint_info['api_key'],
    )

    print("\n✨ Speculative decoding enables 2-3x faster token generation!")
    return result


class DraftModelPipeline:
    """Class for managing draft model creation for speculative decoding."""

    def __init__(self, workspace_ml_client):
        """Initialize with Azure ML workspace client."""
        self.ml_client = workspace_ml_client
        self.guid = str(uuid.uuid4())[:8]

    def create_draft_model_pipeline(
        self,
        registry_ml_client,
        base_model_path,
        training_data_path,
        validation_data_path=None,
        draft_model_config=None,
        compute_name=None,
        num_epochs=1,
        component_name="speculative_decoding_draft_pipeline",
    ):
        """Create and submit draft model training pipeline using registry component."""
        import json

        print("🎯 Creating draft model pipeline...")

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

        print(f"  ✓ Draft config saved: {config_path}")

        # Fetch the pipeline component from registry
        print(f"  ✓ Loading pipeline component: {component_name}")
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
        draft_run = self.ml_client.jobs.create_or_update(
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
        self.ml_client.jobs.download(
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
        import shutil

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
        import json

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

        registered_model = self.ml_client.models.create_or_update(model)
        print(f"  ✓ Model registered: {registered_model.name} v{registered_model.version}")

        return registered_model
