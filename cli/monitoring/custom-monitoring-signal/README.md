# Custom Monitoring Signal

If you are interested in defining your own monitoring signal, metric, and associated threshold, you can do so with AzureML model monitoring.

To do so, start by defining your custom component. The component in this example is defined in `custom-signal-spec.yaml`. The component configuration references the metric definition code found in `src/run.py`.

Register the component to AzureML with this command: `az ml component create -f custom-signal-spec.yaml --subscription <subcription_id> --workspace <workspace_name> --resource-group <resource_group_name>`

After the component has been registered, you can schedule your monitoring job using the configuration found in `custom-monitoring.yaml`.

Schedule your job with the `az ml schedule` command: `az ml schedule create -f custom-monitoring.yaml`