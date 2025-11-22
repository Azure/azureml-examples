import os
import json
import uuid
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from azure.ai.ml import dsl, Input, MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model, Environment, BuildContext
from scripts.run import monitor_run
from scripts.deployment import create_managed_deployment


class DraftModelPipeline:
    """Class for managing draft model creation for speculative decoding."""

    def __init__(self, ml_client: MLClient, registry_ml_client: MLClient):
        self.guid = str(uuid.uuid4())[:8]
        self.ml_client = ml_client
        self.registry_ml_client = registry_ml_client

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
        # Fine-tuning the draft model in speculative decoding makes its predictions closer to the target model, increasing token acceptance
        # and reducing rollbacks. This alignment improves decoding speed and efficiency while maintaining output quality. It also enables
        # better performance for domain-specific tasks by adapting the draft model to relevant data. AzureML provides a prebuilt pipeline for this fine-tuning process.

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

        # Get the draft model pipeline
        try:
            pipeline_component_func = self.registry_ml_client.components.get(
                name=component_name, label="latest"
            )
            print(
                f"Component loaded: {pipeline_component_func.name} v{pipeline_component_func.version}"
            )
        except Exception as e:
            print(f"Failed to load component: {e}")
            print(f"Make sure component '{component_name}' exists in registry")
            raise

        # Define the pipeline job
        @dsl.pipeline
        def create_pipeline():
            draft_pipeline = pipeline_component_func(
                mlflow_model_path=Input(
                    type=AssetTypes.MLFLOW_MODEL, path=base_model_path
                ),
                dataset_train_split=Input(
                    type=AssetTypes.URI_FILE, path=training_data_path
                ),
                dataset_validation_split=Input(
                    type=AssetTypes.URI_FILE, path=validation_data_path
                ),
                draft_model_config=Input(type=AssetTypes.URI_FILE, path=config_path),
                compute_model_import=compute_name,
                compute_eagle3_training=compute_name,
                num_epochs=num_epochs,
            )
            return {"output_model": draft_pipeline.outputs.output_model_path}

        # Create pipeline object
        pipeline_object = create_pipeline()

        # Don't use cached results
        if pipeline_object.settings is not None:
            pipeline_object.settings.force_rerun = True
            pipeline_object.settings.continue_on_step_failure = False

        # Submit job
        print("Submitting draft model pipeline...")
        pipeline_object.display_name = f"draft-model-{self.guid}"
        draft_run = self.ml_client.jobs.create_or_update(
            pipeline_object, experiment_name="speculative-decoding-draft-model"
        )

        print(f"Job submitted: {draft_run.name}")
        print(f"Studio URL: {draft_run.studio_url}")

        return draft_run

    def download_draft_model(self, job_name, output_dir="./models/draft"):
        """Download draft model from completed pipeline job."""
        print(f"Downloading draft model from job: {job_name}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Download model output
        self.ml_client.jobs.download(
            name=job_name,
            output_name="output_model",
            download_path=output_dir,
            all=True,
        )

        print(f"Draft model downloaded to: {output_dir}")

        # Flatten directory structure
        self._flatten_directory(output_dir)

        # Update config with extended context
        self._update_draft_config(output_dir)

        return output_dir

    def _flatten_directory(self, directory):
        """Move all files from subdirectories to root."""

        print("Flattening directory structure...")

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
        # The settings for running a model come both from the model files as well as tuning we apply on top.

        config_path = os.path.join(model_dir, "config.json")

        if not os.path.exists(config_path):
            print("config.json not found, skipping update")
            return

        print("Updating draft model config...")

        with open(config_path, "r") as f:
            draft_config = json.load(f)

        # Update with extended context settings
        draft_config.update(
            {
                "max_position_embeddings": 131072,
                "rope_scaling": {
                    "factor": 8,
                    "high_freq_factor": 4,
                    "low_freq_factor": 1,
                    "original_max_position_embeddings": 8192,
                    "rope_type": "llama3",
                },
            }
        )

        with open(config_path, "w") as f:
            json.dump(draft_config, f, indent=4)

        print("Config updated with extended context settings")

    def upload_combined_model(
        self,
        base_model_dir,
        draft_model_dir,
        model_name="speculative-decoding-combined",
    ):
        """Upload base and draft models as a combined custom model."""
        # A draft model deployment requires both the draft model and the base model.
        # The sglang engine uses the draft model to generate speculative tokens, while the base model
        # verifies and finalizes the output. This function prepares both models for deployment.

        print("Creating combined model package...")
        combined_dir = "./models/"
        print(f"Base model: {base_model_dir}")
        print(f"Draft model: {draft_model_dir}")

        # Register combined model
        model_name_versioned = f"{model_name}-{self.guid}"

        model = Model(
            path=combined_dir,
            name=model_name_versioned,
            description="Combined base and draft model for speculative decoding",
            tags={
                "type": "speculative_decoding",
                "architecture": "eagle3",
            },
        )

        registered_model = self.ml_client.models.create_or_update(model)
        print(f"Model registered: {registered_model.name} v{registered_model.version}")

        return registered_model


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
        "draft_vocab_size": 32000,
    }

    if base_model_config:
        default_config.update(base_model_config)

    return default_config


def run_draft_model_pipeline(
    ml_client: MLClient,
    registry_ml_client: MLClient,
    compute_cluster: str,
    base_model_mlflow_path="azureml://registries/azureml-meta/models/Meta-Llama-3-8B-Instruct/versions/9",
    draft_train_data_path="./data/draft_model/sharegpt_train_small.jsonl",
    num_epochs=1,
    monitor=False,
):
    # Fine-tuning the draft model in speculative decoding makes its predictions closer to the target model, increasing token acceptance
    # and reducing rollbacks. This alignment improves decoding speed and efficiency while maintaining output quality. It also enables
    # better performance for domain-specific tasks by adapting the draft model to relevant data. AzureML provides a prebuilt pipeline for this fine-tuning process.
    print("\n" + "=" * 60)
    print("🎯 STARTING DRAFT MODEL PIPELINE")
    print("=" * 60 + "\n")

    # Create draft model config
    draft_model_config = create_draft_model_config()

    config_dir = "./draft_config"
    os.makedirs(config_dir, exist_ok=True)
    draft_config_path = os.path.join(config_dir, "draft_model_config.json")

    with open(draft_config_path, "w") as f:
        json.dump(draft_model_config, f, indent=4)
    print(f"Draft model config saved: {draft_config_path}")

    # Verify training data
    if not os.path.exists(draft_train_data_path):
        raise FileNotFoundError(
            f"Draft model training data not found: {draft_train_data_path}"
        )
    print(f"Draft training data: {draft_train_data_path}")

    # Get component from registry
    draft_component_name = "eagle3_chat_completion_pipeline"
    eagle3_comp = registry_ml_client.components.get(
        name=draft_component_name, label="latest"
    )

    # Define pipeline
    @dsl.pipeline
    def speculative_decoding_draft_pipeline():
        node = eagle3_comp(
            mlflow_model_path=Input(
                type=AssetTypes.MLFLOW_MODEL, path=base_model_mlflow_path
            ),
            dataset_train_split=Input(
                type=AssetTypes.URI_FILE, path=draft_train_data_path
            ),
            dataset_validation_split=Input(
                type=AssetTypes.URI_FILE, path=draft_train_data_path
            ),
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
    print("Submitting draft model training pipeline...")
    draft_job = ml_client.jobs.create_or_update(
        draft_job, experiment_name="speculative-decoding-draft-model"
    )

    print(f"Job submitted: {draft_job.name}")
    print(f"📊 Studio URL: {draft_job.studio_url}")

    # Monitor if requested
    if monitor:
        _, status = monitor_run(ml_client, draft_job, poll_interval=60)
        return draft_job, status

    return draft_job, None


def prepare_combined_model_for_deployment(
    ml_client: MLClient,
    registry_ml_client: MLClient,
    draft_job_name: str,
    base_model_hf_id="nvidia/Llama-3.1-8B-Instruct-FP8",
    model_name="grpo-speculative-decoding",
    force=False,
):
    # A draft model deployment requires both the draft model and the base model.
    # The sglang engine uses the draft model to generate speculative tokens, while the base model
    # verifies and finalizes the output. This function prepares both models for deployment.
    print("Preparing combined model for deployment...")

    draft_pipeline = DraftModelPipeline(
        ml_client=ml_client, registry_ml_client=registry_ml_client
    )

    # Define paths
    draft_model_dir = "./models/draft"
    base_model_dir = "./models/base"
    base_model_sub_dir = "model_artifact/model"

    print("\nDownloading draft model...")
    ml_client.jobs.download(name=draft_job_name, output_name="output_model", download_path=draft_model_dir, all=True)

    # Move all files from subdirectories to the root directory
    for root, dirs, files in os.walk(draft_model_dir):
        for file in files:
            if root != draft_model_dir:  # Skip files already in root
                source = os.path.join(root, file)
                destination = os.path.join(draft_model_dir, file)            
                shutil.move(source, destination)

    # Remove empty subdirectories
    for root, dirs, files in os.walk(draft_model_dir, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):  # Only remove if empty
                os.rmdir(dir_path)

    # Update draft model config with extended context settings
    draft_config = json.load(open(f"{draft_model_dir}/config.json"))
    draft_config = {
        **draft_config,
        "max_position_embeddings": 131072,
        "rope_scaling": {
            "factor": 8,
            "high_freq_factor": 4,
            "low_freq_factor": 1,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        }
    }
    with open(f"{draft_model_dir}/config.json", "w") as f:
        json.dump(draft_config, f, indent=4)

    # Download base model from HuggingFace
    if force or not os.path.exists(base_model_dir):
        print("\nDownloading base model...")
        snapshot_download(repo_id=base_model_hf_id, local_dir=base_model_dir)
        print(f"Base model downloaded to: {base_model_dir}")

        # Create the target subdirectory structure
        target_dir = os.path.join(base_model_dir, base_model_sub_dir)
        os.makedirs(target_dir, exist_ok=True)
        
        # Move all files and directories from base_model_dir to target_dir
        for item in os.listdir(base_model_dir):
            if item != "model_artifact":  # Skip the subdirectory we just created
                source_path = os.path.join(base_model_dir, item)
                target_path = os.path.join(target_dir, item)
                shutil.move(source_path, target_path)
    else:
        print(f"Base model already exists: {base_model_dir}")

    # Upload combined model
    combined_model = draft_pipeline.upload_combined_model(
        base_model_dir=base_model_dir,
        draft_model_dir=draft_model_dir,
        model_name=model_name,
    )

    print(f"\nCombined model ready for deployment: {combined_model.name}")
    return combined_model


def deploy_speculative_decoding_endpoint(
    ml_client: MLClient,
    combined_model: Model,
    instance_type: str,
):
    print("Deploying speculative decoding endpoint")

    endpoint_name = f"spec-dec-grpo"
    deployment_name = "speculative-deployment"
    model_mount_path = "/var/model-mount"
    endpoint_description = (
        "Speculative decoding endpoint with GRPO fine-tuned base model"
    )
    endpoint_tags = {"model_type": "speculative_decoding", "algorithm": "grpo"}
    environment = Environment(
        build=BuildContext(path="./environment/speculative-decoding-env"),
        name="speculative-decoding-env",
        description="Environment for speculative decoding inference using sglang.",
        inference_config={
            "liveness_route": {
                "port": 30000,
                "path": "/health"
            },
            "readiness_route": {
                "port": 30000,
                "path": "/health_generate"
            },
            "scoring_route": {
                "port": 30000,
                "path": "/"
            }
        }
    )

    environment_variables = {                                                   # Environment variables configure the serving engine and model paths for the container                                         
        "SPECULATIVE_DECODING_MODE": "true",                                    # Used sglang framework for inference
        "BASE_MODEL": f"{model_mount_path}/models/base/model_artifact/model",   # Path for base model
        "DRAFT_MODEL": f"{model_mount_path}/models/draft",                      # Path for draft model
        "NUM_SPECULATIVE_TOKENS": "5",
        "SERVING_ENGINE": "sglang",                                             # the serving engine to use
        "SGLANG_ARGS": "--tp-size 1 --max-running-requests 32 --mem-fraction-static 0.7 --speculative-algorithm EAGLE3 --speculative-num-steps 3 --speculative-eagle-topk 2 --speculative-num-draft-tokens 4 --dtype float16 --decode-attention-backend fa3 --prefill-attention-backend fa3 --host 0.0.0.0 --port 30000 --enable-torch-compile --cuda-graph-max-bs 16",
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        "SGL_HOST": "0.0.0.0",
        "SGL_PORT": "30000",
    }

    endpoint_name = create_managed_deployment(
        ml_client=ml_client,
        model_asset_id=combined_model.id,
        instance_type=instance_type,
        model_mount_path=model_mount_path,
        environment_asset_id=environment,
        endpoint_name=endpoint_name,
        endpoint_description=endpoint_description,
        endpoint_tags=endpoint_tags,
        deployment_name=deployment_name,
        deployment_env_vars=environment_variables,
    )

    print(f"Speculative decoding endpoint deployed: {endpoint_name}")
    return endpoint_name


def deploy_base_model_endpoint(
    ml_client: MLClient,
    instance_type: str,
    base_model="./models/base/model_artifact",
):
    """Deploy base model endpoint."""
    print("Deploying base model endpoint")

    endpoint_name = f"base-model"
    deployment_name = "base-deployment"
    endpoint_description = "Base model endpoint"
    endpoint_tags = {"model_type": "base"}

    model = Model(
        path=base_model,
        name=f"base-model",
    )

    environment_variables = {                                      
        "TASK_TYPE": "chat-completion", 
    }

    endpoint_name = create_managed_deployment(
        ml_client=ml_client,
        model_asset_id=model,
        instance_type=instance_type,
        environment_asset_id=Environment(
            image="mcr.microsoft.com/azureml/curated/foundation-model-inference:85",
            inference_config={
                "liveness_route": {
                    "port": 8000,
                    "path": "/ping"
                },
                "readiness_route": {
                    "port": 8000,
                    "path": "/health"
                },
                "scoring_route": {
                    "port": 8000,
                    "path": "/"
                }
            }
        ),
        endpoint_name=endpoint_name,
        endpoint_description=endpoint_description,
        endpoint_tags=endpoint_tags,
        deployment_name=deployment_name,
        deployment_env_vars=environment_variables,
    )

    print(f"Base model endpoint deployed: {endpoint_name}")
    return endpoint_name


def _prepare_evaluation_inputs(ml_client, base_endpoint_name, speculative_endpoint_name):
    """Prepare necessary inputs for evaluation."""

    print(f"Preparing evaluation inputs...")
    from azure.ai.ml.entities import WorkspaceConnection, ApiKeyConfiguration

    def _get_endpoint_details(endpoint_name):
        """Get endpoint details including scoring URI and API key."""
        api_key = ml_client.online_endpoints.get_keys(name=endpoint_name).primary_key
        base_url = ml_client.online_endpoints.get(name=endpoint_name).scoring_uri.replace("/score", "/")
        connections_name = f"conn-{str(uuid.uuid4())[:8]}"

        ws_connection = WorkspaceConnection(
            name=connections_name,
            type="azure_sql_db",
            target=base_url,
            credentials=ApiKeyConfiguration(key=api_key)
        )
        ml_client.connections.create_or_update(workspace_connection=ws_connection)

        return connections_name, base_url

    base_connection_name, base_scoring_uri = _get_endpoint_details(base_endpoint_name)
    speculative_connection_name, speculative_scoring_uri = _get_endpoint_details(speculative_endpoint_name)
    return base_connection_name, base_scoring_uri, speculative_connection_name, speculative_scoring_uri


def run_evaluation_speculative_decoding(
    ml_client: MLClient,
    registry_ml_client: MLClient,
    base_endpoint_name: str,
    speculative_endpoint_name: str,
    base_model_hf_id: str,
    speculative_model_hf_id: str,
    compute_cluster="h100-dedicated",
    monitor=False,
):
    """Run evaluation pipeline to compare speculative decoding and base model endpoints."""
    base_conn_name, base_score_uri, speculative_conn_name, speculative_score_uri = _prepare_evaluation_inputs(ml_client, base_endpoint_name, speculative_endpoint_name)

    # Load component
    evaluation_component_name = "component_endpoint_benchmarking"
    print(f"  ✓ Loading component: {evaluation_component_name}")
    evaluation_component = registry_ml_client.components.get(name=evaluation_component_name, label="latest")
    print(f"  ✓ Component loaded: {evaluation_component.name} v{evaluation_component.version}")

    # Define pipeline
    @dsl.pipeline(compute=compute_cluster)
    def evaluation_pipeline():
        node = evaluation_component(
            base_scoring_url=base_score_uri,
            base_connection_name=base_conn_name,
            target_scoring_url=speculative_score_uri,
            target_connection_name=speculative_conn_name,
            base_model=base_model_hf_id,
            target_model=speculative_model_hf_id,
            base_backend="vllm",
            target_backend="sglang",
        )
        return {"metrics": node.outputs.metrics}

    # Submit pipeline
    evaluation_job = evaluation_pipeline()
    print("  ✓ Submitting evaluation pipeline...")
    evaluation_job = ml_client.jobs.create_or_update(
        evaluation_job, experiment_name="speculative-decoding-evaluation"
    )

    print(f"Job submitted: {evaluation_job.name}")
    print(f"Studio URL: {evaluation_job.studio_url}")

    # Monitor if requested
    if monitor:
        _, status = monitor_run(ml_client, evaluation_job, poll_interval=60)
        return evaluation_job, status

    return evaluation_job, None
