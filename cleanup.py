from azureml.core import Workspace

ws = Workspace.from_config()

for webservice in ws.webservices:
    ws.webservices[webservice].delete()

for compute_target in ws.compute_targets:
    if "concept" in compute_target:
        ws.compute_targets[compute_target].delete()

for i in range(10000, 99999 + 1):
    try:
        ws2 = Workspace.get(
            f"ws-{i}-concept",
            subscription_id=ws.subscription_id,
            resource_group=ws.resource_group,
        )
        ws2.delete(delete_dependent_resource=True)
    except:
        pass
