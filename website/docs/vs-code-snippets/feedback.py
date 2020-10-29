'''
# Questions

- Include import statements in snippets?
    - e.g.
        ```
        from azureml.core import Workspace
        ws = Workspace.from_config()
        ```
    vs
        ```
        ws = Workspace.from_config
        ```
    - For notebooks this can be useful, for VS code maybe less so?

- More or less?
    - Should we err on the side of including more things, which the user may not need
    or just give the minimal amount e.g.

        ```
        from azureml.core import Workspace, Experiment, ScriptRunConfig

        # get workspace
        ws = Workspace.from_config()

        # get compute target
        target = ws.compute_targets['target-name']

        # get registered environment
        env = ws.environments['env-name']

        # get/create experiment
        exp = Experiment(ws, 'experiment_name')

        # set up script run configuration
        config = ScriptRunConfig(
            source_directory='.',
            script='script.py',
            compute_target=target,
            environment=env,
            arguments=['--meaning', 42],
        )

        # submit script to AML
        run = exp.submit(config)
        print(run.get_portal_url()) # link to ml.azure.com
        run.wait_for_completion(show_output=True)
        ```

    Pros:
        - For new users this shows how pieces fit together and gives useable skeleton
        - Fewer snippets to 'memorize'
    Cons:
        - For experienced users modularity may be preferable

- Workspace as handle vs as input

    ```
    target = ComputeTarget(ws, 'name')
    ```

    OR

    ```
    target = ws.compute_targets['name']
    ```

    Pros:
        - Conceptually cleaner and consistent
        - Throw error right away
    Cons:
        - Slow: calls service

    Similarly for environments, experiments, datastores, datasets, ...
'''

### DEMO

## IMPORTS

# import-core-sdk
# import-*
# ----------------

# ----------------

## Question: one big import vs lots of individual imports?

## GET THINGS

# get-*
# get-env
# ----------------

# ----------------

## Question: include imports? Notebook vs code?
## Question: what's missing

## SCRIPTRUNCONFIG

# script-run-config
# ----------------
from azureml.core import Workspace              # connect to workspace
from azureml.core import Experiment             # connect/create experiments
from azureml.core import ComputeTarget          # connect to compute
from azureml.core import Environment            # manage e.g. Python environments
from azureml.core import Datastore, Dataset     # work with data

from azureml.core import Workspace
ws = Workspace.get(
    name='name'
    subscription_id='subscription_id'
    resource_group='resource_group',
)

# ----------------