from azureml.core import Workspace

ws = Workspace.from_config()

for webservice in ws.webservices:
    ws.webservices[webservice].delete()

for compute_target in ws.compute_targets:
    if "concept" in compute_target:
        ws.compute_targets[compute_target].delete()

workspaces = Workspace.list(ws.subscription_id, resource_group=ws.resource_group)
for workspace in workspaces:
    if "concept" in workspace:
        workspaces[workspace][0].delete(delete_dependent_resource=True, no_wait=True)
