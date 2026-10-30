from __future__ import annotations

from types import SimpleNamespace

import pytest

import foundrytrainingjob.aml_command_job_permissions as permissions
from foundrytrainingjob.aml_command_job_permissions import (
    PermissionCheck,
    PermissionRequirement,
    RoleAssignmentEvidence,
    RuntimePermissionInspection,
    build_runtime_permission_requirements,
    evaluate_permission_requirement,
    grant_missing_reference_storage_access,
    inspect_project_identity_attachment,
    inspect_runtime_permissions,
    list_role_assignments_for_principal,
    scope_covers,
)


_SUBSCRIPTION = "00000000-0000-0000-0000-000000000001"
_PRINCIPAL = "00000000-0000-0000-0000-000000000002"
_RESOURCE_GROUP = f"/subscriptions/{_SUBSCRIPTION}/resourceGroups/rg"
_STORAGE = f"{_RESOURCE_GROUP}/providers/Microsoft.Storage/storageAccounts/storage"
_PROJECT = (
    f"{_RESOURCE_GROUP}/providers/Microsoft.CognitiveServices/accounts/account/"
    "projects/project"
)
_ACR = f"{_RESOURCE_GROUP}/providers/Microsoft.ContainerRegistry/registries/acr"
_UAI = (
    f"{_RESOURCE_GROUP}/providers/Microsoft.ManagedIdentity/"
    "userAssignedIdentities/uai"
)


def _role_id(role_name: str) -> str:
    return (
        f"/subscriptions/{_SUBSCRIPTION}/providers/Microsoft.Authorization/"
        f"roleDefinitions/{permissions.ROLE_IDS[role_name]}"
    )


def _assignment(
    role_name: str,
    scope: str,
    *,
    condition: str | None = None,
) -> RoleAssignmentEvidence:
    return RoleAssignmentEvidence(
        role_definition_id=_role_id(role_name),
        role_name=role_name,
        scope=scope,
        condition=condition,
    )


def _requirement(
    accepted_roles=("Storage Blob Data Contributor",),
) -> PermissionRequirement:
    return PermissionRequirement(
        requirement_id="rbac.storage",
        capability="Storage access",
        scope=_STORAGE,
        accepted_role_names=tuple(accepted_roles),
        reason="read and write blobs",
    )


def test_scope_covers_exact_and_ancestor_but_not_sibling():
    assert scope_covers(_STORAGE, _STORAGE) is True
    assert scope_covers(_RESOURCE_GROUP, _STORAGE) is True
    assert scope_covers(f"/subscriptions/{_SUBSCRIPTION}", _STORAGE) is True
    assert scope_covers(f"{_RESOURCE_GROUP}-other", _STORAGE) is False
    assert scope_covers(_STORAGE + "/blobServices/default", _STORAGE) is False


def test_storage_data_contributor_at_parent_scope_satisfies_requirement():
    check = evaluate_permission_requirement(
        _requirement(),
        (_assignment("Storage Blob Data Contributor", _RESOURCE_GROUP),),
        principal_id=_PRINCIPAL,
    )

    assert check.status == "satisfied"
    assert check.blocking is False
    assert check.observed_assignments[0].scope == _RESOURCE_GROUP


def test_generic_contributor_does_not_substitute_for_blob_data_role():
    generic_contributor = RoleAssignmentEvidence(
        role_definition_id=(
            f"/subscriptions/{_SUBSCRIPTION}/providers/Microsoft.Authorization/"
            "roleDefinitions/b24988ac-6180-42a0-ab88-20f7382dd24c"
        ),
        role_name="Contributor",
        scope=_RESOURCE_GROUP,
    )

    check = evaluate_permission_requirement(
        _requirement(),
        (generic_contributor,),
        principal_id=_PRINCIPAL,
    )

    assert check.status == "missing"
    assert check.blocking is True
    assert check.observed_assignments == (generic_contributor,)


def test_unknown_custom_role_is_unknown_not_falsely_missing():
    custom_role = RoleAssignmentEvidence(
        role_definition_id=(
            f"/subscriptions/{_SUBSCRIPTION}/providers/Microsoft.Authorization/"
            "roleDefinitions/00000000-0000-0000-0000-000000000099"
        ),
        role_name=None,
        scope=_STORAGE,
    )

    check = evaluate_permission_requirement(
        _requirement(),
        (custom_role,),
        principal_id=_PRINCIPAL,
    )

    assert check.status == "unknown"
    assert check.blocking is True
    assert "custom role" in check.message


def test_blob_reader_satisfies_read_but_not_write():
    assignment = _assignment("Storage Blob Data Reader", _STORAGE)

    read_check = evaluate_permission_requirement(
        _requirement(("Storage Blob Data Reader", "Storage Blob Data Contributor")),
        (assignment,),
        principal_id=_PRINCIPAL,
    )
    write_check = evaluate_permission_requirement(
        _requirement(("Storage Blob Data Contributor",)),
        (assignment,),
        principal_id=_PRINCIPAL,
    )

    assert read_check.status == "satisfied"
    assert write_check.status == "missing"


def test_conditional_role_is_not_claimed_unconditionally_satisfied():
    check = evaluate_permission_requirement(
        _requirement(),
        (
            _assignment(
                "Storage Blob Data Contributor",
                _STORAGE,
                condition="@Resource[Microsoft.Storage/storageAccounts:name] StringEquals 'x'",
            ),
        ),
        principal_id=_PRINCIPAL,
    )

    assert check.status == "conditional"
    assert check.blocking is False
    assert "not guaranteed" in check.message


def test_requirement_builder_uses_runtime_specific_roles():
    requirements = build_runtime_permission_requirements(
        foundry_project_id=_PROJECT,
        target_storage_resource_id=_STORAGE,
        source_storage_resource_ids=(_STORAGE, _STORAGE),
        acr_resource_id=_ACR,
    )
    by_id = {item.requirement_id: item for item in requirements}

    assert set(by_id) == {
        "rbac.foundry_project",
        "rbac.target_storage",
        "rbac.source_storage.storage",
        "rbac.environment_acr",
    }
    assert "Foundry User" in by_id["rbac.foundry_project"].accepted_role_names
    assert (
        "Storage Blob Data Contributor"
        in by_id["rbac.target_storage"].accepted_role_names
    )
    assert (
        "Storage Blob Data Reader"
        in by_id["rbac.source_storage.storage"].accepted_role_names
    )
    assert "AcrPull" in by_id["rbac.environment_acr"].accepted_role_names


def test_role_assignment_listing_follows_next_link(monkeypatch):
    pages = iter(
        [
            {
                "value": [
                    {
                        "properties": {
                            "roleDefinitionId": _role_id("Foundry User"),
                            "scope": _PROJECT,
                        }
                    }
                ],
                "nextLink": "https://management.azure.com/next",
            },
            {
                "value": [
                    {
                        "properties": {
                            "roleDefinitionId": _role_id(
                                "Storage Blob Data Contributor"
                            ),
                            "scope": _STORAGE,
                        }
                    }
                ]
            },
        ]
    )
    monkeypatch.setattr(
        permissions,
        "_management_get",
        lambda *args, **kwargs: next(pages),
    )

    result = list_role_assignments_for_principal(
        _SUBSCRIPTION,
        _PRINCIPAL,
        credential=object(),
    )

    assert [item.role_name for item in result] == [
        "Foundry User",
        "Storage Blob Data Contributor",
    ]


def test_role_assignment_listing_rejects_cross_origin_next_link(monkeypatch):
    monkeypatch.setattr(
        permissions,
        "_management_get",
        lambda *args, **kwargs: {
            "value": [],
            "nextLink": "https://attacker.example/roleAssignments",
        },
    )

    with pytest.raises(ValueError, match="management.azure.com origin"):
        list_role_assignments_for_principal(
            _SUBSCRIPTION,
            _PRINCIPAL,
            credential=object(),
        )


def test_resource_listing_rejects_cross_origin_next_link(monkeypatch):
    monkeypatch.setattr(
        permissions,
        "_management_get",
        lambda *args, **kwargs: {
            "value": [],
            "nextLink": "//attacker.example/storageAccounts",
        },
    )

    with pytest.raises(ValueError, match="management.azure.com origin"):
        permissions._list_resources(
            _SUBSCRIPTION,
            provider_path="Microsoft.Storage/storageAccounts",
            api_version="2023-05-01",
            credential=object(),
            timeout_seconds=120,
        )


def test_management_get_resolves_root_relative_url(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured.update(url=url, kwargs=kwargs)
        return SimpleNamespace(json=lambda: {"value": []})

    monkeypatch.setattr(permissions, "get_foundry_json", fake_get)

    permissions._management_get(
        "/subscriptions/sub/providers/Microsoft.Storage/storageAccounts",
        credential=object(),
        timeout_seconds=120,
    )

    assert captured["url"].startswith("https://management.azure.com/")


def test_runtime_inspection_groups_requirements_by_subscription(monkeypatch):
    requirements = build_runtime_permission_requirements(
        foundry_project_id=_PROJECT,
        target_storage_resource_id=_STORAGE,
        source_storage_resource_ids=(),
        acr_resource_id=None,
    )
    list_calls = []
    monkeypatch.setattr(
        permissions,
        "resolve_managed_identity",
        lambda *args, **kwargs: (_PRINCIPAL, "client-id"),
    )

    def fake_list(subscription_id, principal_id, **kwargs):
        list_calls.append((subscription_id, principal_id))
        return (
            _assignment("Foundry User", _PROJECT),
            _assignment("Storage Blob Data Contributor", _STORAGE),
        )

    monkeypatch.setattr(
        permissions,
        "list_role_assignments_for_principal",
        fake_list,
    )

    inspection = inspect_runtime_permissions(
        identity_resource_id=_UAI,
        requirements=requirements,
        credential=object(),
    )

    assert list_calls == [(_SUBSCRIPTION, _PRINCIPAL)]
    assert inspection.principal_id == _PRINCIPAL
    assert {check.status for check in inspection.checks} == {"satisfied"}


def test_role_inspection_failure_is_unknown_not_missing(monkeypatch):
    requirements = (_requirement(),)
    monkeypatch.setattr(
        permissions,
        "resolve_managed_identity",
        lambda *args, **kwargs: (_PRINCIPAL, "client-id"),
    )
    monkeypatch.setattr(
        permissions,
        "list_role_assignments_for_principal",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("403")),
    )

    inspection = inspect_runtime_permissions(
        identity_resource_id=_UAI,
        requirements=requirements,
        credential=object(),
    )

    check = inspection.checks[0]
    assert check.status == "unknown"
    assert check.blocking is True
    assert "not evidence" in check.message
    assert "PermissionError" in str(check.error)


def test_identity_resolution_failure_is_unknown(monkeypatch):
    monkeypatch.setattr(
        permissions,
        "resolve_managed_identity",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("missing")),
    )

    inspection = inspect_runtime_permissions(
        identity_resource_id=_UAI,
        requirements=(_requirement(),),
        credential=object(),
    )

    assert inspection.principal_id is None
    assert inspection.checks[0].status == "unknown"
    assert "ValueError" in str(inspection.checks[0].error)


def test_connection_metadata_extracts_resource_scope_and_aad(monkeypatch):
    class FakeConnections:
        def get(self, **kwargs):
            return {
                "id": _PROJECT + "/connections/storage",
                "name": "storage",
                "type": "AzureStorageAccount",
                "target": "https://storage.blob.core.windows.net/",
                "credentials": {"type": "AAD"},
                "metadata": {"ResourceId": _STORAGE},
            }

    class FakeClient:
        def __init__(self, **kwargs):
            self.connections = FakeConnections()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(
        "azure.ai.projects.AIProjectClient",
        FakeClient,
    )

    result = permissions.inspect_connection_permission_info(
        project_endpoint="https://example.test",
        project_name="project",
        connection_name="storage",
        credential=object(),
    )

    assert result.available is True
    assert result.project_id == _PROJECT
    assert result.resource_id == _STORAGE
    assert result.credential_type == "AAD"


def test_project_identity_attachment_detects_attached_and_missing(monkeypatch):
    monkeypatch.setattr(
        permissions,
        "_management_get",
        lambda *args, **kwargs: {
            "identity": {
                "type": "UserAssigned",
                "userAssignedIdentities": {_UAI: {}},
            }
        },
    )

    attached = inspect_project_identity_attachment(
        _PROJECT,
        _UAI.upper(),
        principal_id=_PRINCIPAL,
        credential=object(),
    )
    missing = inspect_project_identity_attachment(
        _PROJECT,
        f"{_RESOURCE_GROUP}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/other",
        principal_id="other-principal",
        credential=object(),
    )

    assert attached.status == "satisfied"
    assert attached.blocking is False
    assert missing.status == "missing"
    assert missing.blocking is True


def test_blob_and_acr_resource_names_are_parsed():
    assert (
        permissions.storage_account_name_from_uri(
            "https://account.blob.core.windows.net/container/path"
        )
        == "account"
    )
    assert (
        permissions.acr_host_from_image(
            "registry.azurecr.io/repository/image@sha256:abc"
        )
        == "registry.azurecr.io"
    )
    assert permissions.acr_host_from_image("mcr.microsoft.com/image:tag") is None


def test_grants_reader_only_for_missing_reference_storage(monkeypatch):
    captured = {}

    def fake_put(url, payload, **kwargs):
        captured.update(url=url, payload=payload, kwargs=kwargs)
        return SimpleNamespace(status_code=201)

    monkeypatch.setattr(permissions, "put_foundry_json", fake_put)
    missing = PermissionCheck(
        requirement_id="rbac.source_storage.storage",
        capability="Source storage read access",
        status="missing",
        scope=_STORAGE,
        principal_id=_PRINCIPAL,
        accepted_role_names=("Storage Blob Data Reader",),
        reason="Zero-copy input",
        message="Missing",
    )
    project_identity = PermissionCheck(
        requirement_id="rbac.project_identity",
        capability="Target UAI attached to Foundry project",
        status="satisfied",
        scope=_PROJECT,
        principal_id=_PRINCIPAL,
        accepted_role_names=(),
        reason="Foundry runtime identity",
        message="Attached",
    )
    inspection = RuntimePermissionInspection(
        identity_resource_id=_UAI,
        principal_id=_PRINCIPAL,
        client_id="client-id",
        checks=(missing, project_identity),
    )

    grants = grant_missing_reference_storage_access(
        inspection,
        credential=object(),
    )

    assert len(grants) == 1
    grant = grants[0]
    assert grant.scope == _STORAGE
    assert grant.principal_id == _PRINCIPAL
    assert grant.role_name == "Storage Blob Data Reader"
    assert grant.status_code == 201
    assert captured["url"].startswith(
        f"https://management.azure.com{_STORAGE}/providers/"
        "Microsoft.Authorization/roleAssignments/"
    )
    assert captured["payload"] == {
        "properties": {
            "principalId": _PRINCIPAL,
            "principalType": "ServicePrincipal",
            "roleDefinitionId": _role_id("Storage Blob Data Reader"),
        }
    }
    assert captured["kwargs"]["resource_or_scope"] == ("https://management.azure.com")


def test_reference_storage_grant_ignores_satisfied_and_unknown_checks(monkeypatch):
    monkeypatch.setattr(
        permissions,
        "put_foundry_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("No role assignment should be written.")
        ),
    )
    checks = tuple(
        PermissionCheck(
            requirement_id=f"rbac.source_storage.{status}",
            capability="Source storage read access",
            status=status,
            scope=_STORAGE,
            principal_id=_PRINCIPAL,
            accepted_role_names=("Storage Blob Data Reader",),
            reason="Zero-copy input",
            message=status,
        )
        for status in ("satisfied", "conditional", "unknown")
    ) + (
        PermissionCheck(
            requirement_id="rbac.project_identity",
            capability="Target UAI attached to Foundry project",
            status="satisfied",
            scope=_PROJECT,
            principal_id=_PRINCIPAL,
            accepted_role_names=(),
            reason="Foundry runtime identity",
            message="Attached",
        ),
    )

    grants = grant_missing_reference_storage_access(
        RuntimePermissionInspection(_UAI, _PRINCIPAL, "client-id", checks),
        credential=object(),
    )

    assert grants == ()


def test_reference_storage_grant_rejects_child_scope(monkeypatch):
    monkeypatch.setattr(
        permissions,
        "put_foundry_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Invalid scope must be rejected before an ARM write.")
        ),
    )
    check = PermissionCheck(
        requirement_id="rbac.source_storage.storage",
        capability="Source storage read access",
        status="missing",
        scope=f"{_STORAGE}/blobServices/default/containers/data",
        principal_id=_PRINCIPAL,
        accepted_role_names=("Storage Blob Data Reader",),
        reason="Zero-copy input",
        message="Missing",
    )
    project_identity = PermissionCheck(
        requirement_id="rbac.project_identity",
        capability="Target UAI attached to Foundry project",
        status="satisfied",
        scope=_PROJECT,
        principal_id=_PRINCIPAL,
        accepted_role_names=(),
        reason="Foundry runtime identity",
        message="Attached",
    )

    with pytest.raises(ValueError, match="invalid storage-account scope"):
        grant_missing_reference_storage_access(
            RuntimePermissionInspection(
                _UAI,
                _PRINCIPAL,
                "client-id",
                (check, project_identity),
            ),
            credential=object(),
        )


def test_reference_storage_grant_requires_project_attachment():
    missing = PermissionCheck(
        requirement_id="rbac.source_storage.storage",
        capability="Source storage read access",
        status="missing",
        scope=_STORAGE,
        principal_id=_PRINCIPAL,
        accepted_role_names=("Storage Blob Data Reader",),
        reason="Zero-copy input",
        message="Missing",
    )

    with pytest.raises(ValueError, match="verified as attached"):
        grant_missing_reference_storage_access(
            RuntimePermissionInspection(_UAI, _PRINCIPAL, "client-id", (missing,)),
            credential=object(),
        )
