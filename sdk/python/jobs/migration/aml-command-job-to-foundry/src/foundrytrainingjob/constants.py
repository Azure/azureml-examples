from __future__ import annotations

from collections.abc import Mapping
import os
import re
from string import ascii_lowercase, digits
from typing import Any, Final

_PLACEHOLDER_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"<[^<>\s]+>")


def _env_or_default(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _project_scoped_endpoint(project_endpoint: str, project_name: str) -> str:
    if "/api/projects/" in project_endpoint:
        return project_endpoint
    return f"{project_endpoint.rstrip('/')}/api/projects/{project_name}"


def contains_placeholder_token(value: str) -> bool:
    return _PLACEHOLDER_TOKEN_RE.search(value) is not None


def require_concrete_value(
    value: Any,
    *,
    name: str,
    env_var: str | None = None,
) -> str:
    if value is None:
        raise ValueError(f"{name} must be a non-empty string.")
    normalized_value = str(value).strip()
    if not normalized_value:
        raise ValueError(f"{name} must be a non-empty string.")
    if contains_placeholder_token(normalized_value):
        message = f"{name} contains placeholder token(s): {normalized_value!r}."
        if env_var:
            message += f" Set {env_var} to a concrete value or pass {name} explicitly."
        raise ValueError(message)
    return normalized_value


def _child_path(path: str, key: Any) -> str:
    if isinstance(key, str) and key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


def find_placeholder_tokens(value: Any, *, path: str = "$") -> list[tuple[str, str]]:
    if isinstance(value, str):
        return [(path, value)] if contains_placeholder_token(value) else []
    if isinstance(value, Mapping):
        hits: list[tuple[str, str]] = []
        for key, child in value.items():
            hits.extend(find_placeholder_tokens(child, path=_child_path(path, key)))
        return hits
    if isinstance(value, (list, tuple)):
        hits = []
        for index, child in enumerate(value):
            hits.extend(find_placeholder_tokens(child, path=f"{path}[{index}]"))
        return hits
    return []


DEFAULT_API_VERSION: Final[str] = "2026-01-15-preview"
DEFAULT_PROJECT_ENDPOINT: Final[str] = _env_or_default(
    "FOUNDRY_TRAININGJOB__PROJECT_ENDPOINT",
    "FOUNDRY_E2E__PROJECT_ENDPOINT",
    default="https://<account>.services.ai.azure.com",
)
DEFAULT_PROJECT_NAME: Final[str] = _env_or_default(
    "FOUNDRY_TRAININGJOB__PROJECT_NAME",
    "FOUNDRY_E2E__PROJECT_NAME",
    default="<project-name>",
)
DEFAULT_MLFLOW_TRACKING_HOST: Final[str] = _env_or_default(
    "FOUNDRY_TRAININGJOB__MLFLOW_TRACKING_HOST",
    "FOUNDRY_E2E__MLFLOW_TRACKING_HOST",
    default="<mlflow-tracking-host>",
)
DEFAULT_SMI_PROJECT_NAME: Final[str] = "<system-managed-identity-project-name>"
DEFAULT_STORAGE_CONNECTION_NAME: Final[str] = _env_or_default(
    "FOUNDRY_TRAININGJOB__STORAGE_CONNECTION_NAME",
    default="<storage-connection-name>",
)
DEFAULT_SUBSCRIPTION_ID: Final[str] = _env_or_default(
    "FOUNDRY_TRAININGJOB__SUBSCRIPTION_ID",
    "FOUNDRY_E2E__SUBSCRIPTION_ID",
    default="<subscription-id>",
)
DEFAULT_RESOURCE_GROUP: Final[str] = _env_or_default(
    "FOUNDRY_TRAININGJOB__RESOURCE_GROUP",
    "FOUNDRY_E2E__RESOURCE_GROUP",
    default="<resource-group>",
)
DEFAULT_WORKSPACE_NAME: Final[str] = _env_or_default(
    "FOUNDRY_TRAININGJOB__WORKSPACE_NAME",
    "FOUNDRY_E2E__WORKSPACE_NAME",
    default="<workspace-name>",
)
DEFAULT_UMI_PROJECT_ENDPOINT: Final[str] = _env_or_default(
    "FOUNDRY_TRAININGJOB__PROJECT_SCOPED_ENDPOINT",
    "FOUNDRY_TRAININGJOB__UMI_PROJECT_ENDPOINT",
    default=_project_scoped_endpoint(DEFAULT_PROJECT_ENDPOINT, DEFAULT_PROJECT_NAME),
)
DEFAULT_SMI_PROJECT_SCOPED_ENDPOINT: Final[str] = _env_or_default(
    "FOUNDRY_TRAININGJOB__SMI_PROJECT_SCOPED_ENDPOINT",
    "FOUNDRY_TRAININGJOB__SMI_PROJECT_ENDPOINT",
    default=_project_scoped_endpoint(
        DEFAULT_PROJECT_ENDPOINT, DEFAULT_SMI_PROJECT_NAME
    ),
)

DEFAULT_COMMAND: Final[str] = "python -c \"print('hello from Foundry')\""
DEFAULT_CURATED_ENVIRONMENT_ID: Final[
    str
] = "mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:48"
DEFAULT_INPUT_PRINT_ENVIRONMENT_ID: Final[str] = DEFAULT_CURATED_ENVIRONMENT_ID
DEFAULT_SLIME_RL_RAY_ENVIRONMENT_ID: Final[str] = "<environment-image-reference>"
DEFAULT_VERL_RFT_ENVIRONMENT_ID: Final[str] = "<environment-image-reference>"
DEFAULT_NHC_COMPUTE_ENVIRONMENT_IMAGE_REFERENCE: Final[
    str
] = "<environment-image-reference>"
DEFAULT_CUSTOM_ACR_ENVIRONMENT_ID: Final[str] = _env_or_default(
    "FOUNDRY_TRAININGJOB__CUSTOM_ACR_ENVIRONMENT_ID",
    default="<registry>.azurecr.io/<repository>:<tag>",
)
DEFAULT_CURATED_ENVIRONMENT_IMAGE_REFERENCE: Final[str] = DEFAULT_CURATED_ENVIRONMENT_ID
DEFAULT_INPUT_PRINT_ENVIRONMENT_IMAGE_REFERENCE: Final[
    str
] = DEFAULT_INPUT_PRINT_ENVIRONMENT_ID
DEFAULT_SLIME_RL_RAY_ENVIRONMENT_IMAGE_REFERENCE: Final[
    str
] = DEFAULT_SLIME_RL_RAY_ENVIRONMENT_ID
DEFAULT_VERL_RFT_ENVIRONMENT_IMAGE_REFERENCE: Final[
    str
] = DEFAULT_VERL_RFT_ENVIRONMENT_ID
DEFAULT_CUSTOM_ACR_ENVIRONMENT_IMAGE_REFERENCE: Final[
    str
] = DEFAULT_CUSTOM_ACR_ENVIRONMENT_ID

TEST_FOUNDRY_WCUS_CLUSTER_CPU: Final[str] = "<cpu-compute-name>"
TEST_FOUNDRY_WCUS_CLUSTER_GPU: Final[str] = "<gpu-compute-name>"
TEST_FOUNDRY_WCUS_CLUSTER_A100: Final[str] = "<a100-compute-name>"
DEFAULT_COMPUTE_ID: Final[str] = _env_or_default(
    "FOUNDRY_TRAININGJOB__COMPUTE_ID",
    "FOUNDRY_E2E__SCENARIOS__SIMPLE_HELLO_WORLD__COMPUTE_ID",
    default="<foundry-compute-resource-id>",
)
DEFAULT_GPU_COMPUTE_ID: Final[str] = _env_or_default(
    "FOUNDRY_TRAININGJOB__GPU_COMPUTE_ID",
    default="<foundry-gpu-compute-resource-id>",
)
DEFAULT_SMI_COMPUTE_ID: Final[str] = _env_or_default(
    "FOUNDRY_TRAININGJOB__SMI_COMPUTE_ID",
    default="<foundry-smi-compute-resource-id>",
)

DEFAULT_INSTANCE_TYPE: Final[str] = "Singularity.D4_v3"
DEFAULT_INSTANCE_COUNT: Final[int] = 1
DEFAULT_SMI_INSTANCE_TYPE: Final[str] = "Singularity.ND96am_A100_v4-n1"
DEFAULT_GPU_INSTANCE_TYPE: Final[str] = "Singularity.ND96r_H100_v5"
DEFAULT_MULTINODE_GPU_INSTANCE_TYPE: Final[str] = DEFAULT_GPU_INSTANCE_TYPE
DEFAULT_GPU_INSTANCE_COUNT: Final[int] = 1
DEFAULT_GPU_COUNT: Final[int] = 8
DEFAULT_MULTINODE_GPU_COUNT: Final[int] = DEFAULT_GPU_COUNT * 2
DEFAULT_IDENTITY_UAI: Final[str] = _env_or_default(
    "FOUNDRY_TRAININGJOB__IDENTITY_UAI",
    default="<target-uai-resource-id>",
)

DEFAULT_AISUPERCOMPUTER_PROPERTIES: Final[dict[str, str]] = {
    "imageVersion": "",
    "slaTier": "Premium",
    "priority": "high",
}
DEFAULT_ENVIRONMENT_VARIABLES: Final[dict[str, str]] = {}
EXAMPLE_ENVIRONMENT_VARIABLES: Final[dict[str, str]] = {}
DEFAULT_CANARY_TAGS: Final[dict[str, str]] = {}
DEFAULT_CPU_RESOURCES: Final[dict[str, Any]] = {
    "instanceCount": DEFAULT_INSTANCE_COUNT,
    "instanceType": DEFAULT_INSTANCE_TYPE,
    "properties": {"AISuperComputer": DEFAULT_AISUPERCOMPUTER_PROPERTIES},
}
DEFAULT_GPU_RESOURCES: Final[dict[str, Any]] = {
    "instanceCount": DEFAULT_GPU_INSTANCE_COUNT,
    "instanceType": DEFAULT_GPU_INSTANCE_TYPE,
    "properties": {"AISuperComputer": DEFAULT_AISUPERCOMPUTER_PROPERTIES},
}
DEFAULT_COMPUTE_CONFIG_BY_CLUSTER: Final[dict[str, dict[str, Any]]] = {
    TEST_FOUNDRY_WCUS_CLUSTER_CPU: {
        "computeId": DEFAULT_COMPUTE_ID,
        "resources": DEFAULT_CPU_RESOURCES,
    },
    TEST_FOUNDRY_WCUS_CLUSTER_GPU: {
        "computeId": DEFAULT_GPU_COMPUTE_ID,
        "resources": DEFAULT_GPU_RESOURCES,
        "gpuCount": DEFAULT_GPU_COUNT,
    },
    TEST_FOUNDRY_WCUS_CLUSTER_A100: {
        "computeId": DEFAULT_SMI_COMPUTE_ID,
        "resources": {
            "instanceCount": DEFAULT_INSTANCE_COUNT,
            "instanceType": DEFAULT_SMI_INSTANCE_TYPE,
            "properties": {"AISuperComputer": DEFAULT_AISUPERCOMPUTER_PROPERTIES},
        },
        "gpuCount": DEFAULT_GPU_COUNT,
    },
}

FOUNDRY_E2E_LITERAL_KEY: Final[str] = "FOUNDRY_E2E_LITERAL"
FOUNDRY_E2E_FILE_INPUT_KEY: Final[str] = "FOUNDRY_E2E_FILE_INPUT"
FOUNDRY_E2E_FOLDER_INPUT_KEY: Final[str] = "FOUNDRY_E2E_FOLDER_INPUT"
FOUNDRY_E2E_DOWNLOAD_INPUT_KEY: Final[str] = "FOUNDRY_E2E_DOWNLOAD_INPUT"
FOUNDRY_E2E_CODE_INPUT_KEY: Final[str] = "FOUNDRY_E2E_CODE_INPUT"
FOUNDRY_E2E_MODEL_INPUT_KEY: Final[str] = "FOUNDRY_E2E_MODEL_INPUT"
FOUNDRY_E2E_OUTPUT_A_KEY: Final[str] = "FOUNDRY_E2E_OUTPUT_A"
FOUNDRY_E2E_OUTPUT_B_KEY: Final[str] = "FOUNDRY_E2E_OUTPUT_B"
FOUNDRY_E2E_OUTPUT_FILE_KEY: Final[str] = "FOUNDRY_E2E_OUTPUT_FILE"
FOUNDRY_E2E_OUTPUT_DIR_KEY: Final[str] = "FOUNDRY_E2E_OUTPUT_DIR"
FOUNDRY_E2E_MODEL_OUTPUT_KEY: Final[str] = "FOUNDRY_E2E_MODEL_OUTPUT"
FOUNDRY_E2E_METRIC_NAMES_KEY: Final[str] = "FOUNDRY_E2E_METRIC_NAMES"
FOUNDRY_E2E_TIMEOUT_OK_KEY: Final[str] = "FOUNDRY_E2E_TIMEOUT_OK"
FOUNDRY_E2E_MUTATION_KEY: Final[str] = "FOUNDRY_E2E_MUTATION"
FOUNDRY_E2E_PIP_MARKER_KEY: Final[str] = "FOUNDRY_E2E_PIP_MARKER"

INTERACTIVE_ENDPOINTS_ENABLE_KEY: Final[
    str
] = "AZUREML_ENABLE_DEFAULT_INTERACTIVE_ENDPOINTS"
SINGULARITY_SIDECAR_ENABLE_KEY: Final[str] = "_AZUREML_CR_SINGULARITY_ENABLE_SIDECAR"
SINGULARITY_SIDECAR_IMAGE_KEY: Final[str] = "SINGULARITY_SIDECAR_IMAGE_NAME"
SINGULARITY_JOB_UAI_KEY: Final[str] = "_AZUREML_SINGULARITY_JOB_UAI"
JOB_NAME_SUFFIX_ALPHABET: Final[str] = f"{ascii_lowercase}{digits}"
