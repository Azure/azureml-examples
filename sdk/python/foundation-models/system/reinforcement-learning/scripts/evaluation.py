import uuid
from typing import Optional
from azure.ai.ml import dsl, Input, MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Job
from scripts.run import monitor_run

class EvaluationPipeline():
    """Run Evaluation"""

    DEFAULT_CONFIGS = {
        "evaluate_base_model": False,
        "batch_size": 16,
        "temperature": 0.7,
        "top_p": 0.9,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.8,
        "extraction_method": "flexible",
        "number_of_trials": 3,
    }

    def __init__(self, ml_client: MLClient, registry_ml_client: MLClient):
        self.guid = str(uuid.uuid4())[:8]
        self._ml_client = ml_client
        self._eval_pipeline_component = registry_ml_client.components.get(
            name="pipeline_model_evaluation",
            label="latest"
        )

    def create_evaluate_pipeline(
        self,
        compute: str,
        model_dir_1: Input,
        model_dir_2: Input,
        validation_dataset_path: Input,
        base_model_path: Optional[Input] = None,
        instance_type: Optional[str] = None,
        config = {},
    ) -> Job:
        """Create and submit evaluation pipeline job using registry component."""

        # Update default configs with any provided config
        self.DEFAULT_CONFIGS.update(config)
        print(f"Running with config {self.DEFAULT_CONFIGS}")

        @dsl.pipeline
        def create_pipeline():
            eval_pipeline = self._eval_pipeline_component(
                compute=compute,
                instance_type=instance_type,
                base_model_path=base_model_path,
                checkpoint_base_path_1=model_dir_1,
                checkpoint_base_path_2=model_dir_2,
                validation_file=validation_dataset_path,
                **self.DEFAULT_CONFIGS
            )
            return {"evaluation_results": eval_pipeline.outputs.evaluation_results}

        # Create pipeline object
        print("Creating evaluation pipeline...")
        pipeline_object = create_pipeline()

        # Don't use cached results
        if pipeline_object.settings is not None:
            pipeline_object.settings.force_rerun = True
            pipeline_object.settings.continue_on_step_failure = False

        # Submit job
        print("✓ Submitting Model Evaluation Pipeline ...")
        pipeline_object.display_name = f"evaluate-model-{self.guid}"
        eval_run = self._ml_client.jobs.create_or_update(pipeline_object, experiment_name="evaluate-model")

        print(f"✓ Job submitted: {eval_run.name}")
        print(f"📊 Studio URL: {eval_run.studio_url}")

        return eval_run


def run_evaluation_pipeline(
    ml_client: MLClient,
    registry_ml_client: MLClient,
    compute_cluster: str,
    grpo_model_dir: str,
    rlpp_model_dir: str,
    validation_dataset_path: str,
    base_model_path: Optional[str] = None,
    instance_type: Optional[str] = None,
    run_config: dict = {},
):
    """Run evaluation pipeline to compare finetuned models with baseline."""
    print(" Starting Evaluation Pipeline")
    pipeline = EvaluationPipeline(ml_client, registry_ml_client)

    grpo_model_input = Input(type=AssetTypes.URI_FOLDER, path=grpo_model_dir)
    rlpp_model_input = Input(type=AssetTypes.URI_FOLDER, path=rlpp_model_dir)
    base_model_input = Input(type=AssetTypes.URI_FOLDER, path=base_model_path) if isinstance(base_model_path, str) else base_model_path
    validation_dataset_input = Input(type=AssetTypes.URI_FILE, path=validation_dataset_path)

    eval_job = pipeline.create_evaluate_pipeline(
        compute=compute_cluster,
        instance_type=instance_type,
        model_dir_1=grpo_model_input,
        model_dir_2=rlpp_model_input,
        validation_dataset_path=validation_dataset_input,
        base_model_path=base_model_input,
        config=run_config,
    )

    eval_job, status = monitor_run(ml_client, eval_job)
    if status == "Completed":
        print("\n Evaluation completed successfully")
        return eval_job, status
    else:
        print(f"\n Job did not complete successfully: {status}")
        return eval_job, status
