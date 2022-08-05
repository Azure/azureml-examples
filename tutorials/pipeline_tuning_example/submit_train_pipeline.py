import argparse
from dataclasses import dataclass
from pathlib import Path
from azure.ai.ml import dsl, Input, MLClient, load_component

# Authentication package
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

import hydra
from hydra.core.config_store import ConfigStore

@dataclass
class AMLConfig:
    subscription_id: str
    resource_group: str
    workspace: str

@dataclass
class TrainConfig:
    exp_name: str
    data_path: str
    test_train_ratio: float
    learning_rate: float
    n_estimators: int

@dataclass
class PipelineConfig:
    aml_config: AMLConfig
    train_config:TrainConfig

LOCAL_COMPONENTS_DIR = Path(__file__).parent.absolute()

cs = ConfigStore.instance()
cs.store(name="config", node=PipelineConfig)

@hydra.main(config_path="configs", config_name="train_config")
def main(config: PipelineConfig):
    build_and_submit_aml_pipeline(config)

def build_and_submit_aml_pipeline(config):
    """This function can be called from Python
    while the main function is meant for CLI only.
    When calling the main function in Python,
    there is error due to the hydra.main decorator
    """

    if isinstance(config, list):
        with hydra.initialize(config_path="configs"):
            config = hydra.compose(config_name="train_config", overrides=config)
            
    ################################################
    # connect to your Azure ML workspace
    ################################################
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=config.aml_config.subscription_id,
        resource_group_name=config.aml_config.resource_group,
        workspace_name=config.aml_config.workspace,
    )

    ################################################
    # load component functions
    ################################################
    import os
    print("working directory:", os.getcwd())
    data_prep_component = load_component(path=LOCAL_COMPONENTS_DIR / "data_prep" / "data_prep.yaml")
    train_component = load_component(path=LOCAL_COMPONENTS_DIR / "train" / "train.yaml")

    ################################################
    # build pipeline
    ################################################

    @dsl.pipeline(
        compute="cpucluster",
    )
    def credit_defaults_pipeline():
        data_prep_job = data_prep_component(
            # data=pipeline_job_data_input,
            # test_train_ratio=pipeline_job_test_train_ratio,
            data = config.train_config.data_path,
            test_train_ratio = config.train_config.test_train_ratio,
        )

        train_job = train_component(
            train_data=data_prep_job.outputs.train_data,
            test_data=data_prep_job.outputs.test_data,
            learning_rate = config.train_config.learning_rate,
            n_estimators = config.train_config.n_estimators,
        )

        return

    pipeline = credit_defaults_pipeline()

    # submit the pipeline
    run = ml_client.jobs.create_or_update(
        pipeline,
        experiment_name=config.train_config.exp_name,
    )

    # TODO: print the job url
    print("Job url:", run.studio_url)
    
    return run, ml_client


if __name__ == "__main__":
    main()
