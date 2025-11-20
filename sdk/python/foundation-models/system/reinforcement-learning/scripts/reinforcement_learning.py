import uuid
from azure.ai.ml import Input, MLClient, dsl
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from scripts.run import monitor_run


class RLSpecDecPipeline:
    """Main class for managing RL training and Speculative Decoding workflow."""

    def __init__(self, ml_client: MLClient, registry_ml_client: MLClient):
        # We use an unique identifier for naming resources, this prevents name collisions for resources created in this lab
        self.guid = str(uuid.uuid4())[:8]
        self._ml_client = ml_client
        self._registry_ml_client = registry_ml_client

    def create_rl_pipeline(
        self,
        huggingface_id,
        train_data_path,
        valid_data_path,
        compute_cluster,
        config={},
    ):
        """Create and submit RL pipeline job using registry component."""

        # Group Relative Position Optimization (GRPO) and Reinforce Plus Plus (RLPP) are novel Reinforcement techniques
        # designed to finetune a model to comply to a given reward function.  The RL pipeline is an AzureML pipeline which 
        # provides all the steps to finetune a base model using GRPO or RLPP on a given dataset.
        print("Creating RL pipeline...")

        # Use defaults to ensure reproducibility and avoid missing params
        default_config = {
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
        algorithm = default_config.get("algorithm_adv_estimator", "grpo").lower()
        algorithm = algorithm.replace("_", "-")

        # Extract experiment_name from config as that is passed separately
        if "experiment_name" in default_config:
            experiment_name = default_config["experiment_name"]
            del default_config["experiment_name"]
        else:
            experiment_name = f"reinforcement-learning-{algorithm}"

        # Use registry component for versioning and reuse
        pipeline_component_func = self._registry_ml_client.components.get(
            name="pipeline_rl_finetune",
            label="latest"
        )

        # Define pipeline to encapsulate all steps for traceability and reuse
        @dsl.pipeline
        def create_pipeline():
            rl_pipeline = pipeline_component_func(
                huggingface_id=huggingface_id,
                compute_model_import=compute_cluster,
                compute_finetune=compute_cluster,
                data_train_files=Input(type=AssetTypes.URI_FILE, path=train_data_path),
                data_val_files=Input(type=AssetTypes.URI_FILE, path=valid_data_path),
                **default_config, # Pass all config as kwargs for maintainability and future-proofing
            )
            return {"model_output": rl_pipeline.outputs.model_output}

        pipeline_object = create_pipeline()

        # Force rerun to ensure new job, avoid stale results
        if pipeline_object.settings is not None:
            pipeline_object.settings.force_rerun = True
            pipeline_object.settings.continue_on_step_failure = False  # Fail fast for debugging

        # Submit job
        print("Submitting pipeline...")
        pipeline_object.display_name = f"{algorithm}-{self.guid}"
        rl_run = self._ml_client.jobs.create_or_update(pipeline_object, experiment_name=experiment_name)
        print(f"Studio URL: {rl_run.studio_url}")  # Clickable link for monitoring

        return rl_run

    def register_model(self, job, model_name_prefix):
        """Assets must be registered as models before use in endpoints."""
        print("Registering model from job output...")

        # Use GUID to ensure model name uniqueness across runs
        model_name = f"{model_name_prefix}-{self.guid}"
        model_output = job.outputs.model_output

        model = Model(
            path=model_output.path,
            name=model_name,
            description="GRPO fine-tuned model on FinQA",
            type=AssetTypes.CUSTOM_MODEL,
        )

        registered_model = self._ml_client.models.create_or_update(model)  # Register the model
        print(f"Model: {registered_model.name} v{registered_model.version}")  
        print(f"ID: {registered_model.id}")

        return registered_model


def run_rl_training_pipeline(
    ml_client: MLClient,
    registry_ml_client: MLClient,
    base_model_id: str,
    train_data_path: str,
    valid_data_path: str,
    compute_cluster: str,
    rl_method="grpo",
    config={},
):
    # Group Relative Position Optimization (GRPO) and Reinforce Plus Plus (RLPP) are novel Reinforcement techniques
    # designed to finetune a model to comply to a given reward function.  The RL pipeline is an AzureML pipeline which 
    # provides all the steps to finetune a base model using GRPO or RLPP on a given dataset.
    print("Starting RL Training Pipeline")
    pipeline = RLSpecDecPipeline(ml_client, registry_ml_client)

    # # We have uploaded the data assets to our registry in advance for this tutorial
    # train_asset = ml_client.data.get(name="dataset_training_finqa", label="latest")
    # val_asset = ml_client.data.get(name="dataset_validation_finqa", label="latest")

    # Submit RL pipeline job with all required config and assets
    config["algorithm_adv_estimator"] = rl_method
    rl_job = pipeline.create_rl_pipeline(
        huggingface_id=base_model_id,
        train_data_path=train_data_path,
        valid_data_path=valid_data_path,
        compute_cluster=compute_cluster,
        config=config,
    )
    
    completed_job, status = monitor_run(ml_client, rl_job)
    if status == "Completed":
        # Register the trained model for downstream deployment and tracking
        registered_model = pipeline.register_model(
            job=completed_job,
            model_name_prefix="grpo-finqa-model",
            # base_model_id=base_model_id,
        )
        return rl_job, status, registered_model
    else:
        print(f"\n Job did not complete successfully: {status}")
        return rl_job, status, None
