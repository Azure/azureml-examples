import time
import flaml
# from azureml.exceptions import ActivityFailedException
# import azureml.core
# from azureml.core import Run
import submit_train_pipeline
from functools import partial
import os
import logging

logger = logging.getLogger(__name__)

def run_with_config(config: dict):
    """Run the pipeline with a given config dict
    """

    overrides = [f"{key}={value}" for key, value in config.items()]
    # # overwrite the path to deep speed configuration.
    # if isinstance(Run.get_context(), azureml.core.run._OfflineRun):
    #     config_searchpath = os.path.abspath(os.path.join(deepspeed_wd, "..\\.."))
    # else:
    #     config_searchpath = deepspeed_wd
    # overrides += [f'+script_args.deepspeed_wd={deepspeed_wd}', f'hydra.searchpath=[{config_searchpath}]']
    
    print(overrides)
    run = submit_train_pipeline.build_and_submit_aml_pipeline(overrides)
    metrics = {"test_accuracy_score": 0}  # indicate config is bad
    # try:
    #     run.wait_for_completion()  # this line can raise ActivityFailedException
    #     evaluate_run = run._find_child_run("azureml.feed://Babel/babel.sequence_classification.evaluate")
    #     if not evaluate_run:
    #         # using local components
    #         evaluate_run = run._find_child_run("babel.sequence_classification.evaluate")
    #     metrics = evaluate_run[0]._core_run.get_metrics()
    # # except ActivityFailedException:
    # except Exception as error:
    #     # This is an example of pipeline tuning,
    #     # which is different from running a single pipeline.
    #     # In pipeline tuning,
    #     # pipeline runs will be submitted with different hyperparameter configurations.
    #     # If we don't catch the exception,
    #     # one bad hyperparameter configuration which fails the pipeline run
    #     # will terminate the entire pipeline tuning.
    #     # With the catch,
    #     # the failure is still visible to the user as they will see a warning in the console.
    #     print(str(error.exception))
        # metrics = {"test_accuracy_score": 0}  # indicate config is bad
    return metrics

def tune_pipeline(concurrent_run=1):
    start_time = time.time()
    # hyperparameter search space
    search_space = {
        "train_config.n_estimators": flaml.tune.randint(50, 200),
        "train_config.learning_rate": flaml.tune.uniform(0.01, 0.5),
    }
    # # initial points to evaluate
    # points_to_evaluate = [{
    #     "large_model_finetune.max_epochs": 5,
    #     "filter.threshold": 0.8,
    # }]
    HP_METRIC = "test_accuracy_score"
    MODE = "max"
    if concurrent_run > 1:
        import ray  # For parallel tuning

        ray.init(num_cpus=concurrent_run)
        use_ray = True
    else:
        use_ray = False

    # the working directory of the current AML job is '/mnt/azureml/cr/j/somerandomnumber/exe/wd/'
    # the wd contains the file included in the snapshot/code folder.
    # however the implementation in the run_with_config has the working direction as 
    # local_dir + 'ray_results/trail_folder/'
    # need to pass the deepspeed_wd to find the correct file of deepspeed config.
    # tune_wd = os.getcwd()
    analysis = flaml.tune.run(
        run_with_config,
        config=search_space,
        # points_to_evaluate=points_to_evaluate,
        metric=HP_METRIC,
        mode=MODE,
        num_samples=2,  # number of trials
        use_ray=use_ray,
    )
    best_trial = analysis.get_best_trial(HP_METRIC, MODE, "all")
    metric = best_trial.metric_analysis[HP_METRIC][MODE]
    print(f"n_trials={len(analysis.trials)}")
    print(f"time={time.time()-start_time}")
    print(f"Best {HP_METRIC}: {metric:.4f}")
    print(f"Best coonfiguration: {best_trial.config}")


if __name__ == "__main__":
    print("call tune_pipeline correctly.")
    # tune_pipeline(concurrent_run=2)
    # for parallel tuning, pass concurrent_run > 1
