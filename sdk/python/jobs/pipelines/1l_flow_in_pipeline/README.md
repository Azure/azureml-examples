This is a dummy pipeline job with anonymous reference of a flow as a component. This example has reused the flow in corresponding CLI example, which is copied from [sample in promptflow repository](https://github.com/microsoft/promptflow/tree/main/examples/flows/standard/web-classification). Please check [this path](../../../../../cli/jobs/pipelines-with-components/pipeline_job_with_flow_as_component/) for dependent resources.

Note that `.promptflow/flow.tools.json` in the flow directory is required to use a flow as a component. Usually you can use `pf flow validate` or `pf run validate` to generate it.

References:
- [microsoft/promptflow: Build high-quality LLM apps](https://github.com/microsoft/promptflow)
- [Reference - Prompt flow docuentation](https://microsoft.github.io/promptflow/reference/index.html)
