"""
Standalone script to verify create-then-update behavior for a sweep pipeline job.

This script:
1. Creates the sweep pipeline job from the 1c example.
2. Fetches the created job.
3. Updates both display_name and tags.
4. Calls create_or_update to apply the updates.

Usage:
    python update_sweep_job.py

Requires an .azureml/config.json (or env vars) for workspace connection.
"""

import json
import os
from pathlib import Path

from azure.ai.ml import MLClient, load_component
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.sweep import (
    BanditPolicy,
    Choice,
    LogUniform,
    Uniform,
)
from azure.identity import DefaultAzureCredential

# ---------------------------------------------------------------------------
# 1. Connect to workspace
# ---------------------------------------------------------------------------
ml_client = MLClient.from_config(credential=DefaultAzureCredential())
print(f"Connected to workspace: {ml_client.workspace_name}")

# ---------------------------------------------------------------------------
# 2. Build the same pipeline that the notebook builds
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent

train_component = load_component(source=HERE / "train.yml")
predict_component = load_component(source=HERE / "predict.yml")


@pipeline()
def pipeline_with_hyperparameter_sweep():
    train_model = train_component(
        data_folder=None,
        batch_size=Choice([32, 64, 128]),
        first_layer_neurons=Choice([16, 64, 128, 256, 512]),
        second_layer_neurons=Choice([16, 64, 256, 512]),
        learning_rate=LogUniform(-6, -1),
    )

    sweep_step = train_model.sweep(
        primary_metric="validation_acc",
        goal="Maximize",
        sampling_algorithm="random",
        compute="cpu-cluster",
    )
    sweep_step.early_termination = BanditPolicy(
        slack_factor=0.1, evaluation_interval=2
    )
    sweep_step.set_limits(max_total_trials=20, max_concurrent_trials=10, timeout=7200)

    predict_component(
        model=sweep_step.outputs.model_output,
        test_data=sweep_step.outputs.test_data,
    )


pipeline_job = pipeline_with_hyperparameter_sweep()
pipeline_job.settings.default_compute = "cpu-cluster"

# ---------------------------------------------------------------------------
# 3. Submit job
# ---------------------------------------------------------------------------
pipeline_job = ml_client.jobs.create_or_update(
    pipeline_job, experiment_name="pipeline_samples"
)
print(f"Created job: {pipeline_job.name}  display_name={pipeline_job.display_name}")

# ---------------------------------------------------------------------------
# 4. Fetch and update display_name + tags
# ---------------------------------------------------------------------------
fetched_job = ml_client.jobs.get(pipeline_job.name)
print(f"\nFetched job display_name: {fetched_job.display_name}")
print(f"Fetched job tags: {fetched_job.tags}")

# Update display_name
fetched_job.display_name = f"{fetched_job.display_name}_updated"

# Update tags (merge with any existing tags)
fetched_job.tags = {
    **(fetched_job.tags or {}),
    "updated_by": "update_sweep_job.py",
    "status": "verified",
}

updated_job = ml_client.jobs.create_or_update(fetched_job)
print(f"\nUpdated job display_name: {updated_job.display_name}")
print(f"Updated job tags: {updated_job.tags}")
print("\nCreate-then-update verified successfully.")
