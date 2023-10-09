This is a dummy pipeline job with anonymous reference of a flow as a component. Flow directory is copied from [sample in promptflow repository](https://github.com/microsoft/promptflow/tree/main/examples/flows/standard/web-classification).

Note that `.promptflow/flow.tools.json` in the flow directory is required to use a flow as a component. Usually you can use `pf flow validate` or `pf run validate` to generate it.

After that, You need to edit the compute cluster in the defaults section and run `az ml job create --file pipeline.yml` to submit the pipeline job.

References:
- [microsoft/promptflow: Build high-quality LLM apps](https://github.com/microsoft/promptflow)
- [Reference - Prompt flow docuentation](https://microsoft.github.io/promptflow/reference/index.html)
