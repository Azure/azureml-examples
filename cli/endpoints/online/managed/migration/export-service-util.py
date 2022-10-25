import json
import argparse
import tempfile
from azureml.core import Workspace
from azureml.exceptions import WebserviceException
from azureml._model_management._util import _get_mms_url, get_requests_session
from azureml._model_management._constants import (
    AKS_WEBSERVICE_TYPE,
    ACI_WEBSERVICE_TYPE,
    UNKNOWN_WEBSERVICE_TYPE,
    MMS_SYNC_TIMEOUT_SECONDS,
)
from azureml.core.webservice import Webservice, AciWebservice, AksWebservice
from azureml._restclient.clientbase import ClientBase

MIGRATION_WEBSERVICE_TYPES = [AKS_WEBSERVICE_TYPE, ACI_WEBSERVICE_TYPE]


def export(
    ws: Workspace,
    service_name: str = None,
    timeout_seconds: int = None,
    show_output: bool = True,
):
    """
    Export all services under target workspace into template and parameters.
    :param ws: Target workspace.
    :param service_name: the service name to be migrated.
    :param show_output: Whether print outputs.
    :param timeout_seconds: Timeout settings for waiting export.
    """
    base_url = _get_mms_url(ws)
    mms_endpoint = base_url + "/services/" + service_name
    headers = {"Content-Type": "application/json"}
    headers.update(ws._auth_object.get_authentication_header())
    try:
        resp = ClientBase._execute_func(
            get_requests_session().get,
            mms_endpoint,
            headers=headers,
            timeout=MMS_SYNC_TIMEOUT_SECONDS,
        )
    except:
        raise WebserviceException(f"Cannot get service {service_name}")

    if resp.status_code == 404:
        raise WebserviceException(f"Service {service_name} does not exist.")

    content = resp.content
    if isinstance(resp.content, bytes):
        content = resp.content.decode("utf-8")
    service = json.loads(content)
    if service["state"] != "Healthy":
        raise WebserviceException(
            f"service {service_name} is unhealthy, migration with this tool is not supported."
        )
    compute_type = service["computeType"]
    if compute_type.upper() not in MIGRATION_WEBSERVICE_TYPES:
        raise WebserviceException(
            'Invalid compute type "{}". Valid compute types are "{}"'.format(
                compute_type, ",".join(MIGRATION_WEBSERVICE_TYPES)
            )
        )
    compute_name = service_name
    if compute_type.upper() == AKS_WEBSERVICE_TYPE:
        compute_name = service["computeName"]

    mms_endpoint = base_url + "/services/export"
    export_payload = {"serviceName": service_name}
    try:
        resp = ClientBase._execute_func(
            get_requests_session().post,
            mms_endpoint,
            headers=headers,
            json=export_payload,
        )
    except:
        raise WebserviceException(f"Cannot get service {service_name}")

    if resp.status_code == 202:
        service_entity = None
        if compute_type.upper() == AKS_WEBSERVICE_TYPE:
            service_entity = AksWebservice(ws, service_name)
        elif compute_type.upper() == ACI_WEBSERVICE_TYPE:
            service_entity = AciWebservice(ws, service_name)
        service_entity.state = "Exporting"
        service_entity._operation_endpoint = (
            _get_mms_url(service_entity.workspace)
            + f'/operations/{resp.content.decode("utf-8")}'
        )
        state, _, operation = service_entity._wait_for_operation_to_complete(
            show_output, timeout_seconds
        )
        if state == "Succeeded":
            export_folder = operation.get("resourceLocation").split("/")[-1]
            storage_account = service_entity.workspace.get_details().get(
                "storageAccount"
            )
            if show_output:
                print(
                    f"Services have been exported to storage account: {storage_account} \n"
                    f"Folder path: azureml/{export_folder}"
                )
            return storage_account.split("/")[-1], export_folder, compute_name
    else:
        raise WebserviceException(
            "Received bad response from Model Management Service:\n"
            "Response Code: {}\n"
            "Headers: {}\n"
            "Content: {}".format(resp.status_code, resp.headers, resp.content)
        )


def overwrite_parameters(
    parms: dict, endpoint_name: str = None, deployment_name: str = None
):
    """
    Overwrite parameters
    :param deployment_name: v2 online-deployment name. Default will be v1 service name.
    :param endpoint_name: v2 online-endpoint name. Default will be v1 service name.
    :param parms: parameters as dict: loaded v2 parameters.
    """
    properties = parms["onlineEndpointProperties"]["value"]
    traffic = parms["onlineEndpointPropertiesTrafficUpdate"]["value"]
    properties.pop("keys")
    traffic.pop("keys")
    if endpoint_name:
        parms["onlineEndpointName"]["value"] = endpoint_name

    # this is optional
    if deployment_name:
        parms["onlineDeployments"]["value"][0]["name"] = deployment_name
        traffic["traffic"][deployment_name] = traffic["traffic"].pop(
            list(traffic["traffic"].keys())[0]
        )

    temp_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False)
    json.dump(online_endpoint_deployment, temp_file)
    temp_file.flush()
    print(temp_file.name)


if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser(description="Export v1 service script")
        parser.add_argument(
            "--export", action="store_true", help="using script for export services"
        )
        parser.add_argument(
            "--overwrite-parameters",
            action="store_true",
            help="using script for overwrite parameters purpose",
        )
        parser.add_argument("-w", "--workspace", type=str, help="workspace name")
        parser.add_argument(
            "-g", "--resource-group", type=str, help="resource group name"
        )
        parser.add_argument("-s", "--subscription", type=str, help="subscription id")
        parser.add_argument(
            "-sn",
            "--service-name",
            default=None,
            type=str,
            help="service name to be migrated",
        )
        parser.add_argument(
            "-e",
            "--export-json",
            action="store_true",
            dest="export_json",
            help="show export result in json",
        )
        parser.add_argument(
            "-mp", "--parameters-path", type=str, help="parameters file path"
        )
        parser.add_argument(
            "-me",
            "--migrate-endpoint-name",
            type=str,
            default=None,
            help="v2 online-endpoint name, default is v1 service name",
        )
        parser.add_argument(
            "-md",
            "--migrate-deployment-name",
            type=str,
            default=None,
            help="v2 online-deployment name, default is v1 service name",
        )
        parser.set_defaults(compute_type=None)
        return parser.parse_args()

    # parse args
    args = parse_args()

    if args.export:
        workspace = Workspace.get(
            name=args.workspace,
            resource_group=args.resource_group,
            subscription_id=args.subscription,
        )
        storage_account, blob_folder, v1_compute = export(
            workspace, args.service_name, show_output=not args.export_json
        )
        if args.export_json:
            print(
                json.dumps(
                    {
                        "storage_account": storage_account,
                        "blob_folder": blob_folder,
                        "v1_compute": v1_compute,
                    }
                )
            )

    if args.overwrite_parameters:
        with open(args.parameters_path) as f:
            online_endpoint_deployment = json.load(f)
        overwrite_parameters(
            online_endpoint_deployment,
            args.migrate_endpoint_name,
            args.migrate_deployment_name,
        )
