---
title: Workspace
---

Workspaces are a foundational object used throughout Azure ML and are used in the
constructors of many other classes. Throughout this documentation we frequently
omit the workspace object instantiation and simply refer to `ws`.

See [Installation](installation) for instructions on creating a new workspace.

## Get workspace

Instantiate `Workspace` object used to connect to your AML assets.

```python title="run.py"
from azureml.core import Workspace
ws = Workspace(
    subscription_id="<subscription_id>",
    resource_group="<resource_group>",
    workspace_name="<workspace_name>",
)
```

For convenience store your workspace metadata in a `config.json`.

```json title=".azureml/config.json"
{
    "subscription_id": <subscription-id>,
    "resource_group": <resource-group>,
    "workspace_name": <workspace-name>
}
```

### Helpful methods

- `ws.write_config(path, file_name)` : Write the `config.json` on your behalf. The `path` defaults to '.azureml/' in the current working directory and `file_name` defaults to 'config.json'.
- `Workspace.from_config(path, _file_name)`: Read the workspace configuration from config. The parameter defaults to starting the search in the current directory.

:::info
It is recommended to store these in a directory `.azureml/` as this path is searched _by default_
in the `Workspace.from_config` method.
:::

## Get Workspace Assets

The workspace provides a handle to your Azure ML assets:

### Compute Targets

Get all compute targets attached to the workspace.

```python
ws.compute_targets: Dict[str, ComputeTarget]
```

### Datastores

Get all datastores registered to the workspace.

```python
ws.datastores: Dict[str, Datastore]
```

Get the workspace's default datastore.

```python
ws.get_default_datastore(): Datastore
```

### Keyvault

Get workspace's default Keyvault.

```python
ws.get_default_keyvault(): Keyvault
```

### Environments

Get environments registered to the workspace.

```python
ws.environments: Dict[str, Environment]
```

### MLFlow

Get MLFlow tracking uri.

```python
ws.get_mlflow_tracking_uri(): str
```