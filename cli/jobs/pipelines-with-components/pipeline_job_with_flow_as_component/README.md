This is a dummy pipeline job with anonymous reference of a flow as a component. Flow directory is copied from [sample in promptflow repository](https://github.com/microsoft/promptflow/tree/main/examples/flows/standard/basic) and remove connection dependency to avoid using promptflow connection in azure ml example repository.

Prerequirements:
1. `.promptflow/flow.tools.json` in the flow directory is required to use a flow as a component. Usually you can use `pf flow validate` or `pf run validate` to generate it.
2. You should either update connection name in `flow.dag.yaml` or update `connection.yaml` with your own api information and use `pf connection create --file connection.yaml` to create a workspace connection.
3. You need to either edit the compute cluster in `pipeline.yml` or create a compute cluster named `cpu-cluster` in your workspace.
4. Please ensure that there are `$schema` in your `flow.dag.yaml` and `run.yaml`
    1. `flow.dag.yaml`: `$schema: https://azuremlschemas.azureedge.net/promptflow/latest/Flow.schema.json`
    2. `run.yaml`: `$schema: https://azuremlschemas.azureedge.net/promptflow/latest/Run.schema.json`

After that, you can run `az ml job create --file pipeline.yml` to submit the pipeline job.

References:
- [microsoft/promptflow: Build high-quality LLM apps](https://github.com/microsoft/promptflow)
- [Reference - Prompt flow docuentation](https://microsoft.github.io/promptflow/reference/index.html)
