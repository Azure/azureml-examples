# SA v1 Azure ML workflow

This folder contains a local Azure ML pipeline workflow composed of:

- `step_1_component.yml`
- `pipeline_component.yml`
- `workflow.yml`

## Submit the workflow

From the repo root, run:

```bash
az ml job create --file cli/jobs/pipelines-with-components/pipeline/workflow.yml
```

## Notes

- Update `settings.default_compute` in `workflow.yml` to a compute cluster that exists in your workspace.
- If you prefer serverless, remove `default_compute` and set serverless settings as needed.
