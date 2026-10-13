"""Runtime identity inspection and opt-in RBAC grants for AML job migration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urlsplit
from uuid import NAMESPACE_URL, uuid5

from azure.core.credentials import TokenCredential

from .dataset import build_project_endpoint
from .rest import DEFAULT_TIMEOUT_SECONDS, get_foundry_json, put_foundry_json


_MANAGEMENT_RESOURCE = "https://management.azure.com"
_IDENTITY_API_VERSION = "2023-01-31"
_AUTHORIZATION_API_VERSION = "2022-04-01"
_STORAGE_API_VERSION = "2023-05-01"
_ACR_API_VERSION = "2023-07-01"
_COGNITIVE_SERVICES_API_VERSION = "2025-06-01"

ROLE_IDS: dict[str, str] = {
    "Storage Blob Data Reader": "2a2b9908-6ea1-4ae2-8e65-a410df84e7d1",
    "Storage Blob Data Contributor": "ba92f5b4-2d11-453d-a403-e96b0029c9fe",
    "Storage Blob Data Owner": "b7e6dc6d-f1e8-4753-8033-0f276bb0955b",
    "AcrPull": "7f951dda-4ed3-4680-a7ca-43fe172d538d",
    "AcrPush": "8311e382-0749-4cb8-b61a-304f252e45ec",
    "Container Registry Repository Reader": "b93aa761-3e63-49ed-ac28-beffa264f7ac",
    "Container Registry Repository Writer": "2a1e307c-b015-4ebd-883e-5b7698a07328",
    "Container Registry Repository Contributor": "2efddaa5-3f1f-4df3-97df-af3f13818f4c",
    "Foundry User": "53ca6127-db72-4b80-b1b0-d745d6d5456d",
    "Azure AI Developer": "64702f94-c441-49e6-a78b-ef80e0188fee",
    "Azure AI Administrator": "b78c5d69-af96-48a3-bf8d-a8b4d589de94",
}

_STORAGE_READ_ROLES = (
    "Storage Blob Data Reader",
    "Storage Blob Data Contributor",
    "Storage Blob Data Owner",
)
_STORAGE_WRITE_ROLES = (
    "Storage Blob Data Contributor",
    "Storage Blob Data Owner",
)
_ACR_PULL_ROLES = (
    "AcrPull",
    "AcrPush",
    "Container Registry Repository Reader",
    "Container Registry Repository Writer",
    "Container Registry Repository Contributor",
)
_FOUNDRY_PROJECT_ROLES = (
    "Foundry User",
    "Azure AI Developer",
    "Azure AI Administrator",
)


def _management_url(url: str) -> str:
    candidate = str(url).strip()
    if not candidate:
        raise ValueError("ARM request URL must be non-empty.")

    parsed = urlsplit(candidate)
    if not parsed.scheme and not parsed.netloc:
        if not candidate.startswith("/"):
            raise ValueError("ARM relative request URLs must start with '/'.")
        candidate = f"{_MANAGEMENT_RESOURCE}{candidate}"
        parsed = urlsplit(candidate)

    expected = urlsplit(_MANAGEMENT_RESOURCE)
    if (
        parsed.scheme.lower() != expected.scheme.lower()
        or parsed.netloc.lower() != expected.netloc.lower()
    ):
        raise ValueError(
            f"ARM request URLs must use the {_MANAGEMENT_RESOURCE} origin."
        )
    return candidate


@dataclass(frozen=True)
class ConnectionPermissionInfo:
    """Non-secret metadata needed to evaluate a project connection."""

    name: str
    available: bool
    connection_id: str | None = None
    connection_type: str | None = None
    target: str | None = None
    resource_id: str | None = None
    credential_type: str | None = None
    error: str | None = None

    @property
    def project_id(self) -> str | None:
        marker = "/connections/"
        if self.connection_id and marker in self.connection_id.lower():
            index = self.connection_id.lower().index(marker)
            return self.connection_id[:index]
        return None


@dataclass(frozen=True)
class RoleAssignmentEvidence:
    role_definition_id: str
    role_name: str | None
    scope: str
    condition: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "roleDefinitionId": self.role_definition_id,
            "roleName": self.role_name,
            "scope": self.scope,
            "condition": self.condition,
        }


@dataclass(frozen=True)
class PermissionRequirement:
    requirement_id: str
    capability: str
    scope: str
    accepted_role_names: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class PermissionCheck:
    requirement_id: str
    capability: str
    status: str
    scope: str | None
    principal_id: str | None
    accepted_role_names: tuple[str, ...]
    reason: str
    message: str
    observed_assignments: tuple[RoleAssignmentEvidence, ...] = ()
    error: str | None = None

    @property
    def blocking(self) -> bool:
        return self.status in {"missing", "unknown"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.requirement_id,
            "capability": self.capability,
            "status": self.status,
            "scope": self.scope,
            "principalId": self.principal_id,
            "acceptedRoleNames": list(self.accepted_role_names),
            "reason": self.reason,
            "message": self.message,
            "blocking": self.blocking,
            "observedAssignments": [
                assignment.to_dict() for assignment in self.observed_assignments
            ],
            "error": self.error,
        }


@dataclass(frozen=True)
class RuntimePermissionInspection:
    identity_resource_id: str | None
    principal_id: str | None
    client_id: str | None
    checks: tuple[PermissionCheck, ...]
    limitations: tuple[str, ...] = (
        "Role-assignment inspection does not evaluate deny assignments, PIM-eligible but inactive roles, network ACLs, private DNS, or service-side connection policy.",
        "Caller permissions to create Foundry assets/jobs and AML export jobs are not proven by this runtime-UAI check.",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "identityResourceId": self.identity_resource_id,
            "principalId": self.principal_id,
            "clientId": self.client_id,
            "checks": [check.to_dict() for check in self.checks],
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class ReferenceStorageRoleGrant:
    requirement_id: str
    identity_resource_id: str
    principal_id: str
    scope: str
    role_name: str
    role_definition_id: str
    role_assignment_id: str
    status_code: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirementId": self.requirement_id,
            "identityResourceId": self.identity_resource_id,
            "principalId": self.principal_id,
            "scope": self.scope,
            "roleName": self.role_name,
            "roleDefinitionId": self.role_definition_id,
            "roleAssignmentId": self.role_assignment_id,
            "statusCode": self.status_code,
        }


def _subscription_id(resource_id: str | None) -> str | None:
    if not resource_id:
        return None
    parts = str(resource_id).strip("/").split("/")
    for index, part in enumerate(parts[:-1]):
        if part.lower() == "subscriptions":
            return parts[index + 1]
    return None


def subscription_id_from_resource_id(resource_id: str | None) -> str | None:
    return _subscription_id(resource_id)


def _resource_name_from_blob_uri(uri: str | None) -> str | None:
    if not uri:
        return None
    hostname = urlsplit(uri).hostname or ""
    marker = ".blob."
    if marker not in hostname.lower():
        return None
    return hostname[: hostname.lower().index(marker)]


def _registry_host(image_reference: str | None) -> str | None:
    if not image_reference:
        return None
    first_segment = str(image_reference).split("/", 1)[0].lower()
    return first_segment if ".azurecr.io" in first_segment else None


def _role_guid(role_definition_id: str | None) -> str:
    return str(role_definition_id or "").rstrip("/").rsplit("/", 1)[-1].lower()


def _role_name(role_definition_id: str | None) -> str | None:
    guid = _role_guid(role_definition_id)
    return next(
        (name for name, role_id in ROLE_IDS.items() if role_id.lower() == guid),
        None,
    )


def _normalize_scope(scope: str | None) -> str:
    return str(scope or "").rstrip("/").lower()


def scope_covers(assignment_scope: str, required_scope: str) -> bool:
    """Return whether an assignment at a parent scope applies to a child scope."""

    assignment = _normalize_scope(assignment_scope)
    required = _normalize_scope(required_scope)
    return bool(assignment and required) and (
        required == assignment or required.startswith(assignment + "/")
    )


def _json_object(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _management_get(
    url: str,
    *,
    credential: TokenCredential,
    timeout_seconds: float,
) -> dict[str, Any]:
    response = get_foundry_json(
        _management_url(url),
        resource_or_scope=_MANAGEMENT_RESOURCE,
        credential=credential,
        timeout_seconds=timeout_seconds,
    )
    return _json_object(response.json())


def inspect_connection_permission_info(
    *,
    project_endpoint: str,
    project_name: str,
    connection_name: str,
    credential: TokenCredential,
) -> ConnectionPermissionInfo:
    """Read non-secret connection metadata used by RBAC analysis."""

    try:
        from azure.ai.projects import AIProjectClient

        endpoint = build_project_endpoint(
            project_endpoint,
            project_name=project_name,
        )
        with AIProjectClient(endpoint=endpoint, credential=credential) as client:
            connection = client.connections.get(
                name=connection_name,
                include_credentials=False,
            )
        payload = dict(connection) if isinstance(connection, Mapping) else {}
        metadata = payload.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        credentials = payload.get("credentials")
        credential_type = None
        if isinstance(credentials, Mapping):
            credential_type = str(credentials.get("type") or "") or None
        elif credentials:
            text = str(credentials)
            for candidate in ("AAD", "ApiKey", "SAS", "ManagedIdentity"):
                if candidate.lower() in text.lower():
                    credential_type = candidate
                    break
        return ConnectionPermissionInfo(
            name=connection_name,
            available=True,
            connection_id=str(payload.get("id") or "") or None,
            connection_type=str(payload.get("type") or "") or None,
            target=str(payload.get("target") or "") or None,
            resource_id=str(metadata.get("ResourceId") or "") or None,
            credential_type=credential_type,
        )
    except Exception as error:
        return ConnectionPermissionInfo(
            name=connection_name,
            available=False,
            error=f"{type(error).__name__}: {error}",
        )


def resolve_managed_identity(
    identity_resource_id: str,
    *,
    credential: TokenCredential,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[str, str | None]:
    query = urlencode({"api-version": _IDENTITY_API_VERSION})
    payload = _management_get(
        f"{_MANAGEMENT_RESOURCE}{identity_resource_id}?{query}",
        credential=credential,
        timeout_seconds=timeout_seconds,
    )
    properties = _json_object(payload.get("properties"))
    principal_id = str(properties.get("principalId") or "")
    if not principal_id:
        raise ValueError(
            f"Managed identity {identity_resource_id!r} exposed no principalId."
        )
    client_id = str(properties.get("clientId") or "") or None
    return principal_id, client_id


def inspect_project_identity_attachment(
    project_resource_id: str,
    identity_resource_id: str,
    *,
    principal_id: str | None,
    credential: TokenCredential,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> PermissionCheck:
    """Verify that a target UAI is attached to the Foundry project resource."""

    requirement_id = "rbac.project_identity"
    capability = "Target UAI attached to Foundry project"
    reason = "Foundry command jobs can use only identities attached to the project."
    try:
        query = urlencode({"api-version": _COGNITIVE_SERVICES_API_VERSION})
        payload = _management_get(
            f"{_MANAGEMENT_RESOURCE}{project_resource_id}?{query}",
            credential=credential,
            timeout_seconds=timeout_seconds,
        )
        identity = _json_object(payload.get("identity"))
        assigned = _json_object(identity.get("userAssignedIdentities"))
        attached = _normalize_scope(identity_resource_id) in {
            _normalize_scope(resource_id) for resource_id in assigned
        }
        return PermissionCheck(
            requirement_id=requirement_id,
            capability=capability,
            status="satisfied" if attached else "missing",
            scope=project_resource_id,
            principal_id=principal_id,
            accepted_role_names=(),
            reason=reason,
            message=(
                "The target UAI is attached to the Foundry project."
                if attached
                else "The target UAI is not attached to the Foundry project."
            ),
        )
    except Exception as error:
        return PermissionCheck(
            requirement_id=requirement_id,
            capability=capability,
            status="unknown",
            scope=project_resource_id,
            principal_id=principal_id,
            accepted_role_names=(),
            reason=reason,
            message="The Foundry project's attached identities could not be inspected.",
            error=f"{type(error).__name__}: {error}",
        )


def list_role_assignments_for_principal(
    subscription_id: str,
    principal_id: str,
    *,
    credential: TokenCredential,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[RoleAssignmentEvidence, ...]:
    query = urlencode(
        {
            "api-version": _AUTHORIZATION_API_VERSION,
            "$filter": f"principalId eq '{principal_id}'",
        }
    )
    url: str | None = (
        f"{_MANAGEMENT_RESOURCE}/subscriptions/{subscription_id}/providers/"
        f"Microsoft.Authorization/roleAssignments?{query}"
    )
    assignments: list[RoleAssignmentEvidence] = []
    while url:
        payload = _management_get(
            url,
            credential=credential,
            timeout_seconds=timeout_seconds,
        )
        values = payload.get("value")
        for value in values if isinstance(values, list) else ():
            item = _json_object(value)
            properties = _json_object(item.get("properties"))
            role_definition_id = str(properties.get("roleDefinitionId") or "")
            scope = str(properties.get("scope") or "")
            if not role_definition_id or not scope:
                continue
            assignments.append(
                RoleAssignmentEvidence(
                    role_definition_id=role_definition_id,
                    role_name=_role_name(role_definition_id),
                    scope=scope,
                    condition=(
                        str(properties.get("condition"))
                        if properties.get("condition")
                        else None
                    ),
                )
            )
        next_link = payload.get("nextLink")
        url = _management_url(str(next_link)) if next_link else None
    return tuple(assignments)


def evaluate_permission_requirement(
    requirement: PermissionRequirement,
    assignments: Sequence[RoleAssignmentEvidence],
    *,
    principal_id: str,
) -> PermissionCheck:
    accepted_ids = {
        ROLE_IDS[name].lower()
        for name in requirement.accepted_role_names
        if name in ROLE_IDS
    }
    matching = tuple(
        assignment
        for assignment in assignments
        if scope_covers(assignment.scope, requirement.scope)
        and _role_guid(assignment.role_definition_id) in accepted_ids
    )
    unconditional = tuple(item for item in matching if not item.condition)
    if unconditional:
        return PermissionCheck(
            requirement_id=requirement.requirement_id,
            capability=requirement.capability,
            status="satisfied",
            scope=requirement.scope,
            principal_id=principal_id,
            accepted_role_names=requirement.accepted_role_names,
            reason=requirement.reason,
            message="A required effective role assignment is present.",
            observed_assignments=matching,
        )
    if matching:
        return PermissionCheck(
            requirement_id=requirement.requirement_id,
            capability=requirement.capability,
            status="conditional",
            scope=requirement.scope,
            principal_id=principal_id,
            accepted_role_names=requirement.accepted_role_names,
            reason=requirement.reason,
            message=(
                "A matching assignment exists but has an Azure RBAC condition; "
                "runtime access is not guaranteed by static analysis."
            ),
            observed_assignments=matching,
        )
    observed_at_scope = tuple(
        assignment
        for assignment in assignments
        if scope_covers(assignment.scope, requirement.scope)
    )
    unknown_custom_roles = tuple(
        assignment for assignment in observed_at_scope if assignment.role_name is None
    )
    if unknown_custom_roles:
        return PermissionCheck(
            requirement_id=requirement.requirement_id,
            capability=requirement.capability,
            status="unknown",
            scope=requirement.scope,
            principal_id=principal_id,
            accepted_role_names=requirement.accepted_role_names,
            reason=requirement.reason,
            message=(
                "A custom role assignment exists at an effective scope, but its "
                "actions/dataActions were not resolved by static analysis."
            ),
            observed_assignments=observed_at_scope,
        )
    return PermissionCheck(
        requirement_id=requirement.requirement_id,
        capability=requirement.capability,
        status="missing",
        scope=requirement.scope,
        principal_id=principal_id,
        accepted_role_names=requirement.accepted_role_names,
        reason=requirement.reason,
        message=(
            "No accepted effective role assignment was found at the required "
            "scope or an ancestor scope."
        ),
        observed_assignments=observed_at_scope,
    )


def _list_resources(
    subscription_id: str,
    *,
    provider_path: str,
    api_version: str,
    credential: TokenCredential,
    timeout_seconds: float,
) -> tuple[dict[str, Any], ...]:
    query = urlencode({"api-version": api_version})
    url: str | None = (
        f"{_MANAGEMENT_RESOURCE}/subscriptions/{subscription_id}/providers/"
        f"{provider_path}?{query}"
    )
    result: list[dict[str, Any]] = []
    while url:
        payload = _management_get(
            url,
            credential=credential,
            timeout_seconds=timeout_seconds,
        )
        values = payload.get("value")
        if isinstance(values, list):
            result.extend(dict(item) for item in values if isinstance(item, Mapping))
        next_link = payload.get("nextLink")
        url = _management_url(str(next_link)) if next_link else None
    return tuple(result)


def resolve_storage_account_resource_id(
    account_name: str,
    subscription_ids: Sequence[str],
    *,
    credential: TokenCredential,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> str | None:
    target = account_name.lower()
    for subscription_id in dict.fromkeys(subscription_ids):
        for account in _list_resources(
            subscription_id,
            provider_path="Microsoft.Storage/storageAccounts",
            api_version=_STORAGE_API_VERSION,
            credential=credential,
            timeout_seconds=timeout_seconds,
        ):
            if str(account.get("name") or "").lower() == target:
                return str(account.get("id") or "") or None
    return None


def resolve_acr_resource_id(
    registry_host: str,
    subscription_ids: Sequence[str],
    *,
    credential: TokenCredential,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> str | None:
    target = registry_host.lower()
    for subscription_id in dict.fromkeys(subscription_ids):
        for registry in _list_resources(
            subscription_id,
            provider_path="Microsoft.ContainerRegistry/registries",
            api_version=_ACR_API_VERSION,
            credential=credential,
            timeout_seconds=timeout_seconds,
        ):
            properties = _json_object(registry.get("properties"))
            login_server = str(properties.get("loginServer") or "").lower()
            if login_server == target:
                return str(registry.get("id") or "") or None
    return None


def build_runtime_permission_requirements(
    *,
    foundry_project_id: str | None,
    target_storage_resource_id: str | None,
    source_storage_resource_ids: Sequence[str],
    acr_resource_id: str | None,
) -> tuple[PermissionRequirement, ...]:
    requirements: list[PermissionRequirement] = []
    if foundry_project_id:
        requirements.append(
            PermissionRequirement(
                requirement_id="rbac.foundry_project",
                capability="Foundry project runtime access",
                scope=foundry_project_id,
                accepted_role_names=_FOUNDRY_PROJECT_ROLES,
                reason="The job identity must access the target Foundry project.",
            )
        )
    if target_storage_resource_id:
        requirements.append(
            PermissionRequirement(
                requirement_id="rbac.target_storage",
                capability="Foundry output storage data-plane access",
                scope=target_storage_resource_id,
                accepted_role_names=_STORAGE_WRITE_ROLES,
                reason=(
                    "The job identity must read project inputs and write outputs "
                    "without storage account keys."
                ),
            )
        )
    for resource_id in dict.fromkeys(source_storage_resource_ids):
        suffix = resource_id.rstrip("/").rsplit("/", 1)[-1]
        requirements.append(
            PermissionRequirement(
                requirement_id=f"rbac.source_storage.{suffix}",
                capability=f"Source storage read access ({suffix})",
                scope=resource_id,
                accepted_role_names=_STORAGE_READ_ROLES,
                reason=(
                    "Zero-copy Dataset V3 inputs require the target identity to "
                    "read the AML source blobs."
                ),
            )
        )
    if acr_resource_id:
        requirements.append(
            PermissionRequirement(
                requirement_id="rbac.environment_acr",
                capability="Private ACR image pull",
                scope=acr_resource_id,
                accepted_role_names=_ACR_PULL_ROLES,
                reason="The job identity must pull the referenced private image.",
            )
        )
    return tuple(requirements)


def inspect_runtime_permissions(
    *,
    identity_resource_id: str | None,
    requirements: Sequence[PermissionRequirement],
    credential: TokenCredential,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> RuntimePermissionInspection:
    if not identity_resource_id:
        checks = tuple(
            PermissionCheck(
                requirement_id=requirement.requirement_id,
                capability=requirement.capability,
                status="unknown",
                scope=requirement.scope,
                principal_id=None,
                accepted_role_names=requirement.accepted_role_names,
                reason=requirement.reason,
                message="No target user-assigned identity resource ID was supplied.",
            )
            for requirement in requirements
        )
        return RuntimePermissionInspection(None, None, None, checks)
    try:
        principal_id, client_id = resolve_managed_identity(
            identity_resource_id,
            credential=credential,
            timeout_seconds=timeout_seconds,
        )
    except Exception as error:
        checks = tuple(
            PermissionCheck(
                requirement_id=requirement.requirement_id,
                capability=requirement.capability,
                status="unknown",
                scope=requirement.scope,
                principal_id=None,
                accepted_role_names=requirement.accepted_role_names,
                reason=requirement.reason,
                message="The target identity could not be resolved.",
                error=f"{type(error).__name__}: {error}",
            )
            for requirement in requirements
        )
        return RuntimePermissionInspection(
            identity_resource_id,
            None,
            None,
            checks,
        )

    assignments_by_subscription: dict[
        str, tuple[RoleAssignmentEvidence, ...] | Exception
    ] = {}
    for requirement in requirements:
        subscription_id = _subscription_id(requirement.scope)
        if not subscription_id or subscription_id in assignments_by_subscription:
            continue
        try:
            assignments_by_subscription[
                subscription_id
            ] = list_role_assignments_for_principal(
                subscription_id,
                principal_id,
                credential=credential,
                timeout_seconds=timeout_seconds,
            )
        except Exception as error:
            assignments_by_subscription[subscription_id] = error

    checks: list[PermissionCheck] = []
    for requirement in requirements:
        subscription_id = _subscription_id(requirement.scope)
        assignments_or_error = assignments_by_subscription.get(
            str(subscription_id),
            ValueError(f"Could not derive subscription from {requirement.scope!r}."),
        )
        if isinstance(assignments_or_error, Exception):
            checks.append(
                PermissionCheck(
                    requirement_id=requirement.requirement_id,
                    capability=requirement.capability,
                    status="unknown",
                    scope=requirement.scope,
                    principal_id=principal_id,
                    accepted_role_names=requirement.accepted_role_names,
                    reason=requirement.reason,
                    message=(
                        "Role assignments could not be inspected. This is not "
                        "evidence that the target role is missing."
                    ),
                    error=(
                        f"{type(assignments_or_error).__name__}: "
                        f"{assignments_or_error}"
                    ),
                )
            )
            continue
        checks.append(
            evaluate_permission_requirement(
                requirement,
                assignments_or_error,
                principal_id=principal_id,
            )
        )
    return RuntimePermissionInspection(
        identity_resource_id,
        principal_id,
        client_id,
        tuple(checks),
    )


def grant_missing_reference_storage_access(
    inspection: RuntimePermissionInspection,
    *,
    credential: TokenCredential,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[ReferenceStorageRoleGrant, ...]:
    """Grant Blob Data Reader for analyzer-confirmed missing reference scopes."""

    identity_resource_id = str(inspection.identity_resource_id or "").strip()
    principal_id = str(inspection.principal_id or "").strip()
    if not identity_resource_id or not principal_id:
        raise ValueError(
            "The target user-assigned identity must resolve before reference "
            "storage access can be granted."
        )
    project_identity_checks = tuple(
        check
        for check in inspection.checks
        if check.requirement_id == "rbac.project_identity"
    )
    if (
        len(project_identity_checks) != 1
        or project_identity_checks[0].status != "satisfied"
    ):
        raise ValueError(
            "The target user-assigned identity must be verified as attached "
            "to the Foundry project before reference storage access can be "
            "granted."
        )

    role_name = "Storage Blob Data Reader"
    role_id = ROLE_IDS[role_name]
    grants: list[ReferenceStorageRoleGrant] = []
    for check in inspection.checks:
        if not (
            check.requirement_id.startswith("rbac.source_storage.")
            and check.status == "missing"
        ):
            continue
        scope = str(check.scope or "").strip().rstrip("/")
        subscription_id = _subscription_id(scope)
        scope_parts = scope.strip("/").split("/")
        if (
            not subscription_id
            or len(scope_parts) != 8
            or scope_parts[0].lower() != "subscriptions"
            or scope_parts[2].lower() != "resourcegroups"
            or scope_parts[4].lower() != "providers"
            or scope_parts[5].lower() != "microsoft.storage"
            or scope_parts[6].lower() != "storageaccounts"
            or not scope_parts[7]
        ):
            raise ValueError(
                f"Source-storage requirement {check.requirement_id!r} has an "
                f"invalid storage-account scope: {scope!r}."
            )
        if check.principal_id and check.principal_id.lower() != principal_id.lower():
            raise ValueError(
                f"Source-storage requirement {check.requirement_id!r} belongs "
                "to a different principal."
            )

        role_definition_id = (
            f"/subscriptions/{subscription_id}/providers/"
            f"Microsoft.Authorization/roleDefinitions/{role_id}"
        )
        assignment_name = str(
            uuid5(
                NAMESPACE_URL,
                "|".join(
                    (
                        "aml-foundry-migrate",
                        scope.lower(),
                        principal_id.lower(),
                        role_id,
                    )
                ),
            )
        )
        role_assignment_id = (
            f"{scope}/providers/Microsoft.Authorization/"
            f"roleAssignments/{assignment_name}"
        )
        query = urlencode({"api-version": _AUTHORIZATION_API_VERSION})
        response = put_foundry_json(
            f"{_MANAGEMENT_RESOURCE}{role_assignment_id}?{query}",
            {
                "properties": {
                    "principalId": principal_id,
                    "principalType": "ServicePrincipal",
                    "roleDefinitionId": role_definition_id,
                }
            },
            resource_or_scope=_MANAGEMENT_RESOURCE,
            credential=credential,
            timeout_seconds=timeout_seconds,
        )
        grants.append(
            ReferenceStorageRoleGrant(
                requirement_id=check.requirement_id,
                identity_resource_id=identity_resource_id,
                principal_id=principal_id,
                scope=scope,
                role_name=role_name,
                role_definition_id=role_definition_id,
                role_assignment_id=role_assignment_id,
                status_code=response.status_code,
            )
        )
    return tuple(grants)


def storage_account_name_from_uri(uri: str | None) -> str | None:
    return _resource_name_from_blob_uri(uri)


def acr_host_from_image(image_reference: str | None) -> str | None:
    return _registry_host(image_reference)
