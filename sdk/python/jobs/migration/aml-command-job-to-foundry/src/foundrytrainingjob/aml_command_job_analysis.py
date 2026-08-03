"""Read-only compatibility analysis for AML command-job migration."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .aml_command_job_migration import audit_aml_command_job_compatibility
from .aml_command_job_migrator import (
    AmlWorkspace,
    FoundryTarget,
    materialize_aml_command_job,
    parse_aml_asset_reference,
    _resolve_aml_datastore_uri,
)
from .dataset import build_project_endpoint
from .aml_command_job_permissions import (
    ConnectionPermissionInfo,
    PermissionCheck,
    RuntimePermissionInspection,
    acr_host_from_image,
    build_runtime_permission_requirements,
    inspect_connection_permission_info,
    inspect_project_identity_attachment,
    inspect_runtime_permissions,
    resolve_acr_resource_id,
    resolve_storage_account_resource_id,
    storage_account_name_from_uri,
    subscription_id_from_resource_id,
)


ANALYSIS_POLICIES = (
    "advisory",
    "migratable",
    "lossless",
    "reference-only",
    "strict",
)
_LITERAL_TYPES = frozenset({"literal", "string", "integer", "number", "boolean"})
_DATA_TYPES = frozenset({"uri_file", "uri_folder", "mltable"})
_MODEL_TYPES = frozenset({"custom_model", "mlflow_model", "triton_model"})
_INPUT_MODES = frozenset(
    {"download", "direct", "eval_download", "eval_mount", "ro_mount", "rw_mount"}
)
_OUTPUT_MODES = frozenset({"direct", "rw_mount", "upload"})
_INPUT_MODE_TARGETS = {
    "download": "Download",
    "direct": "Direct",
    "eval_download": "EvalDownload",
    "eval_mount": "EvalMount",
    "ro_mount": "ReadOnlyMount",
    "rw_mount": "ReadWriteMount",
}
_OUTPUT_MODE_TARGETS = {
    "direct": "Direct",
    "rw_mount": "ReadWriteMount",
    "upload": "ReadWriteMount",
}
_DISTRIBUTIONS = frozenset({"mpi", "pytorch", "tensorflow", "ray"})
_GENERATED_SERVICES = frozenset({"local", "studio", "tracking"})
_SUPPORTED_SERVICES = frozenset(
    {
        "jupyter",
        "jupyterlab",
        "ssh",
        "tensorboard",
        "vscode",
        "theia",
        "grafana",
        "custom",
    }
)
_CAPABILITY_MATRIX_TEST = "tests/unit/test_aml_command_job_capability_matrix.py"


@dataclass(frozen=True)
class CapabilityFamilyContract:
    """Required test ownership and known live evidence for a capability family."""

    description: str
    unit_test: str = _CAPABILITY_MATRIX_TEST
    live_evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "unitTest": self.unit_test,
            "liveEvidence": list(self.live_evidence),
        }


CAPABILITY_FAMILY_CATALOG: dict[str, CapabilityFamilyContract] = {
    "job.type": CapabilityFamilyContract(
        "Standalone command-job discriminator.",
        live_evidence=("tests/e2e/test_aml_command_job_migration.py",),
    ),
    "job.command": CapabilityFamilyContract(
        "Command text and placeholder adaptations.",
        live_evidence=("tests/e2e/test_aml_command_job_migration.py",),
    ),
    "asset.code": CapabilityFamilyContract(
        "Code snapshot migration into Dataset V3 codeId.",
        live_evidence=(
            "tests/e2e/test_aml_command_job_migration.py",
            "scenario:code_id",
        ),
    ),
    "asset.environment": CapabilityFamilyContract(
        "OCI environment image reuse or build requirement.",
        live_evidence=(
            "tests/e2e/test_aml_command_job_migration.py",
            "scenario:environment_custom_acr",
        ),
    ),
    "job.compute": CapabilityFamilyContract(
        "Foundry compute and instance-type replacement.",
        live_evidence=("tests/e2e/test_aml_command_job_migration.py",),
    ),
    "job.identity": CapabilityFamilyContract(
        "Target UAI replacement and runtime identity.",
        live_evidence=(
            "tests/e2e/test_aml_command_job_migration.py",
            "scenario:identity_runtime_assertion",
        ),
    ),
    "connection.source_storage": CapabilityFamilyContract(
        "Foundry connection for zero-copy source data.",
        live_evidence=("tests/e2e/test_aml_command_job_migration.py",),
    ),
    "input.binding": CapabilityFamilyContract(
        "Literal, data, and model input bindings.",
        live_evidence=(
            "tests/e2e/test_aml_command_job_migration.py",
            "scenario:input_content_round_trip",
            "scenario:input_model_and_data",
        ),
    ),
    "input.definition": CapabilityFamilyContract(
        "Optional/default/range/enum input-definition metadata."
    ),
    "input.mode": CapabilityFamilyContract(
        "Input delivery modes and unsupported-mode rejection.",
        live_evidence=(
            "scenario:input_mode_download",
            "scenario:input_public_url",
        ),
    ),
    "input.metadata": CapabilityFamilyContract(
        "Fixed compute path, datastore, and IP metadata."
    ),
    "output.binding": CapabilityFamilyContract(
        "Data and model output allocation/registration.",
        live_evidence=(
            "tests/e2e/test_aml_command_job_migration.py",
            "scenario:output_content_round_trip",
            "scenario:output_custom_model",
        ),
    ),
    "output.mode": CapabilityFamilyContract(
        "Output delivery modes and unsupported-mode rejection.",
        live_evidence=(
            "scenario:output_file",
            "scenario:output_folder",
        ),
    ),
    "output.early_available": CapabilityFamilyContract(
        "AML early-available output semantics."
    ),
    "job.environment_variables": CapabilityFamilyContract(
        "Static and placeholder-templated environment variables.",
        live_evidence=("tests/e2e/test_aml_command_job_migration.py",),
    ),
    "job.resources": CapabilityFamilyContract(
        "Instance count, shared memory, and multi-node resources.",
        live_evidence=(
            "tests/e2e/test_aml_command_job_migration.py",
            "scenario:resources_multinode_pytorch",
        ),
    ),
    "job.resources.aml_only": CapabilityFamilyContract(
        "AML-only Docker arguments and location hints."
    ),
    "job.distribution": CapabilityFamilyContract(
        "MPI, PyTorch, TensorFlow, and Ray distributions.",
        live_evidence=(
            "scenario:resources_multinode_pytorch",
            "scenario:slime_ray_lifecycle",
        ),
    ),
    "job.limits.timeout": CapabilityFamilyContract(
        "Command timeout translation.",
        live_evidence=(
            "tests/e2e/test_aml_command_job_migration.py",
            "scenario:resources_timeout",
        ),
    ),
    "service.binding": CapabilityFamilyContract(
        "Generated and interactive job services."
    ),
    "job.scheduling": CapabilityFamilyContract(
        "Queue tier, priority, and Foundry SLA adaptation."
    ),
    "job.properties": CapabilityFamilyContract(
        "Portable custom properties and runtime-property filtering.",
        live_evidence=("scenario:slime_ray_lifecycle",),
    ),
    "job.control_plane_metadata": CapabilityFamilyContract(
        "Parent lineage, notifications, IP, and parameter metadata."
    ),
    "job.is_deterministic": CapabilityFamilyContract(
        "AML deterministic/reuse semantics."
    ),
    "rbac.permission": CapabilityFamilyContract(
        "Target runtime identity resolution and effective Azure RBAC roles.",
        unit_test="tests/unit/test_aml_command_job_permissions.py",
        live_evidence=(
            "tests/e2e/test_project_provisioning_flow.py",
            "scenario:identity_runtime_assertion",
        ),
    ),
}


def capability_family_for_id(capability_id: str) -> str:
    """Map a concrete finding ID to its catalog family or fail closed."""

    if capability_id in CAPABILITY_FAMILY_CATALOG:
        return capability_id
    if capability_id.startswith("input."):
        if capability_id.endswith(".definition"):
            return "input.definition"
        if capability_id.endswith(".mode"):
            return "input.mode"
        if capability_id.endswith(".metadata"):
            return "input.metadata"
        return "input.binding"
    if capability_id.startswith("output."):
        if capability_id.endswith(".early_available"):
            return "output.early_available"
        if capability_id.endswith(".mode"):
            return "output.mode"
        return "output.binding"
    if capability_id.startswith("service."):
        return "service.binding"
    if capability_id.startswith("rbac."):
        return "rbac.permission"
    raise ValueError(f"Capability {capability_id!r} has no test-owned catalog family.")


@dataclass(frozen=True)
class CapabilityAssessment:
    """One source capability and its selected migration disposition."""

    capability_id: str
    family: str
    path: str
    category: str
    capability: str
    support: str
    semantic_fidelity: str
    selected_action: str
    verification: str
    message: str
    remediation: str | None = None
    blocking: bool = False
    external_dependency: bool = False
    referenceable: bool | None = None
    source_value: Any = None
    target_value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.capability_id,
            "family": self.family,
            "path": self.path,
            "category": self.category,
            "capability": self.capability,
            "support": self.support,
            "semanticFidelity": self.semantic_fidelity,
            "selectedAction": self.selected_action,
            "verification": self.verification,
            "message": self.message,
            "remediation": self.remediation,
            "blocking": self.blocking,
            "externalDependency": self.external_dependency,
            "referenceable": self.referenceable,
            "sourceValue": self.source_value,
            "targetValue": self.target_value,
            "testCoverage": CAPABILITY_FAMILY_CATALOG[self.family].to_dict(),
        }


@dataclass(frozen=True)
class EnvironmentInspection:
    source_reference: str | None
    image_reference: str | None
    issue: str | None = None
    has_conda_overlay: bool = False
    has_build_context: bool = False


@dataclass(frozen=True)
class MigrationAnalysisReport:
    """Structured, policy-evaluable analysis with no migration side effects."""

    source_job_name: str
    source_job_type: str
    policy: str
    dataset_transfer_mode: str
    environment: EnvironmentInspection
    capabilities: tuple[CapabilityAssessment, ...]
    compatibility_warnings: tuple[str, ...]
    permission_inspection: RuntimePermissionInspection | None = None

    @property
    def summary(self) -> dict[str, Any]:
        blocking = tuple(item for item in self.capabilities if item.blocking)
        lossy = tuple(
            item
            for item in self.capabilities
            if item.semantic_fidelity in {"lossy", "unknown"}
        )
        dependencies = tuple(
            item for item in self.capabilities if item.external_dependency
        )
        not_referenceable = tuple(
            item for item in dependencies if item.referenceable is not True
        )
        not_live_verified = tuple(
            item
            for item in self.capabilities
            if item.verification not in {"live", "not_applicable"}
        )
        can_migrate = not blocking
        lossless = can_migrate and not lossy
        can_reference_all = can_migrate and not not_referenceable
        all_live_verified = can_migrate and not not_live_verified
        encountered_families = tuple(
            sorted({item.family for item in self.capabilities})
        )
        families_without_live_evidence = tuple(
            family
            for family in encountered_families
            if not CAPABILITY_FAMILY_CATALOG[family].live_evidence
        )
        policy_results = {
            "advisory": True,
            "migratable": can_migrate,
            "lossless": lossless,
            "reference-only": can_reference_all,
            "strict": lossless and all_live_verified,
        }
        permission_findings = tuple(
            item for item in self.capabilities if item.family == "rbac.permission"
        )
        permission_statuses = Counter(
            str(item.source_value or "unknown") for item in permission_findings
        )
        return {
            "canMigrateCurrentInvocation": can_migrate,
            "canMigrateDefinitionLosslessly": lossless,
            "canReferenceAllDependencies": can_reference_all,
            "allCapabilitiesLiveVerified": all_live_verified,
            "encounteredCapabilityFamilies": list(encountered_families),
            "capabilityFamilyCount": len(encountered_families),
            "unitTestCoverageComplete": all(
                bool(CAPABILITY_FAMILY_CATALOG[family].unit_test)
                for family in encountered_families
            ),
            "familiesWithoutLiveEvidence": list(families_without_live_evidence),
            "runtimePermissionsSatisfied": bool(permission_findings)
            and all(
                str(item.source_value) == "satisfied" for item in permission_findings
            ),
            "permissionCountsByStatus": dict(sorted(permission_statuses.items())),
            "missingPermissionIds": [
                item.capability_id
                for item in permission_findings
                if str(item.source_value) == "missing"
            ],
            "unknownPermissionIds": [
                item.capability_id
                for item in permission_findings
                if str(item.source_value) == "unknown"
            ],
            "conditionalPermissionIds": [
                item.capability_id
                for item in permission_findings
                if str(item.source_value) == "conditional"
            ],
            "requiresManualAction": any(
                item.support == "manual_action" for item in self.capabilities
            ),
            "policy": self.policy,
            "policyPassed": policy_results[self.policy],
            "countsBySupport": dict(
                sorted(Counter(item.support for item in self.capabilities).items())
            ),
            "countsByAction": dict(
                sorted(
                    Counter(item.selected_action for item in self.capabilities).items()
                )
            ),
            "countsBySemanticFidelity": dict(
                sorted(
                    Counter(
                        item.semantic_fidelity for item in self.capabilities
                    ).items()
                )
            ),
            "countsByVerification": dict(
                sorted(Counter(item.verification for item in self.capabilities).items())
            ),
            "blockingCapabilityIds": [item.capability_id for item in blocking],
            "lossyOrUnknownCapabilityIds": [item.capability_id for item in lossy],
            "nonReferenceableDependencyIds": [
                item.capability_id for item in not_referenceable
            ],
            "notLiveVerifiedCapabilityIds": [
                item.capability_id for item in not_live_verified
            ],
        }

    @property
    def policy_passed(self) -> bool:
        return bool(self.summary["policyPassed"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "source": {
                "jobName": self.source_job_name,
                "jobType": self.source_job_type,
            },
            "strategy": {
                "datasetTransferMode": self.dataset_transfer_mode,
                "environmentTransferMode": "reference",
                "codeTransferMode": "copy",
                "modelTransferMode": "copy",
            },
            "environment": {
                "sourceReference": self.environment.source_reference,
                "imageReference": self.environment.image_reference,
                "issue": self.environment.issue,
                "hasCondaOverlay": self.environment.has_conda_overlay,
                "hasBuildContext": self.environment.has_build_context,
            },
            "summary": self.summary,
            "capabilities": [item.to_dict() for item in self.capabilities],
            "capabilityCatalog": {
                family: contract.to_dict()
                for family, contract in CAPABILITY_FAMILY_CATALOG.items()
            },
            "permissions": (
                self.permission_inspection.to_dict()
                if self.permission_inspection is not None
                else None
            ),
            "compatibilityWarnings": list(self.compatibility_warnings),
        }


def _value(source: Any, name: str) -> Any:
    if isinstance(source, Mapping) and name in source:
        return source.get(name)
    try:
        return getattr(source, name, None)
    except Exception:
        return None


def _reference_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("id", "path"):
            if value.get(key):
                return str(value[key])
    for name in ("id", "path"):
        candidate = _value(value, name)
        if candidate:
            return str(candidate)
    return str(value)


def _client_for_reference(
    reference: Any,
    *,
    source: AmlWorkspace,
    credential: Any,
    ml_client: Any,
) -> Any:
    if not all(
        (
            reference.subscription_id,
            reference.resource_group,
            reference.workspace_name,
        )
    ):
        return ml_client
    if (
        str(reference.subscription_id).lower() == source.subscription_id.lower()
        and str(reference.resource_group).lower() == source.resource_group.lower()
        and str(reference.workspace_name).lower() == source.workspace_name.lower()
    ):
        return ml_client
    from azure.ai.ml import MLClient

    return MLClient(
        credential,
        reference.subscription_id,
        reference.resource_group,
        reference.workspace_name,
    )


def _inspect_environment(
    source_entity: Any,
    *,
    source: AmlWorkspace,
    credential: Any,
    ml_client: Any,
    override: str | None,
) -> EnvironmentInspection:
    source_value = _value(source_entity, "environment")
    source_reference = _reference_text(source_value)
    if override:
        return EnvironmentInspection(
            source_reference=source_reference,
            image_reference=str(override),
        )
    reference = parse_aml_asset_reference(
        source_reference,
        expected_kind="environment",
    )
    if reference is None:
        image = _value(source_value, "image")
        if image:
            return EnvironmentInspection(
                source_reference=source_reference,
                image_reference=str(image),
                has_conda_overlay=bool(_value(source_value, "conda_file")),
                has_build_context=bool(_value(source_value, "build")),
                issue=(
                    "AML environment has a Conda overlay or build context and "
                    "requires a prebuilt image override."
                    if _value(source_value, "conda_file")
                    or _value(source_value, "build")
                    else None
                ),
            )
        if source_reference and not source_reference.lower().startswith("azureml:"):
            return EnvironmentInspection(
                source_reference=source_reference,
                image_reference=source_reference,
            )
        return EnvironmentInspection(
            source_reference=source_reference,
            image_reference=None,
            issue="Could not resolve the AML environment to a reusable image.",
        )
    try:
        client = _client_for_reference(
            reference,
            source=source,
            credential=credential,
            ml_client=ml_client,
        )
        environment = client.environments.get(
            reference.name,
            version=reference.version,
            label=reference.label,
        )
        has_conda = bool(_value(environment, "conda_file"))
        has_build = bool(_value(environment, "build"))
        image = _value(environment, "image")
        issue = None
        if has_conda or has_build:
            issue = (
                "AML Conda overlays and Docker build contexts require a prebuilt "
                "image supplied with --environment-image."
            )
        elif not image:
            issue = "AML environment exposes no reusable image reference."
        return EnvironmentInspection(
            source_reference=source_reference,
            image_reference=str(image) if image else None,
            issue=issue,
            has_conda_overlay=has_conda,
            has_build_context=has_build,
        )
    except Exception as error:
        return EnvironmentInspection(
            source_reference=source_reference,
            image_reference=None,
            issue=f"Environment lookup failed: {type(error).__name__}: {error}",
        )


def _inspect_connection(
    target: FoundryTarget,
    connection_name: str | None,
    *,
    credential: Any,
) -> tuple[bool | None, str | None]:
    if not connection_name:
        return None, "No Foundry source-storage connection was supplied."
    try:
        from azure.ai.projects import AIProjectClient

        endpoint = build_project_endpoint(
            target.project_endpoint,
            project_name=target.project_name,
        )
        with AIProjectClient(endpoint=endpoint, credential=credential) as client:
            connection = client.connections.get(
                name=connection_name,
                include_credentials=False,
            )
        return True, str(_value(connection, "target") or "") or None
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def _resolve_data_reference_uri(
    source_uri: str,
    *,
    source: AmlWorkspace,
    credential: Any,
    ml_client: Any,
) -> tuple[str | None, str | None]:
    try:
        reference = parse_aml_asset_reference(source_uri, expected_kind="data")
        client = ml_client
        data_path = source_uri
        if reference is not None:
            client = _client_for_reference(
                reference,
                source=source,
                credential=credential,
                ml_client=ml_client,
            )
            data = client.data.get(
                reference.name,
                version=reference.version,
                label=reference.label,
            )
            data_path = _reference_text(_value(data, "path")) or ""
        resolved = _resolve_aml_datastore_uri(client, data_path)
        if urlsplit(resolved).scheme.lower() != "https":
            return None, f"Resolved URI is not HTTPS: {resolved!r}."
        return resolved, None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def analyze_materialized_aml_command_job(
    aml_job: Mapping[str, Any],
    *,
    policy: str = "advisory",
    dataset_transfer_mode: str = "upload",
    source_storage_connection_name: str | None = None,
    source_storage_connection_available: bool | None = None,
    source_storage_connection_detail: str | None = None,
    environment: EnvironmentInspection | None = None,
    user_assigned_identity_id: str | None = None,
    source_code_path: str | Path | None = None,
    data_reference_uris: Mapping[str, str | None] | None = None,
    data_reference_errors: Mapping[str, str | None] | None = None,
    permission_inspection: RuntimePermissionInspection | None = None,
) -> MigrationAnalysisReport:
    """Analyze a materialized command job without creating assets or jobs."""

    if policy not in ANALYSIS_POLICIES:
        raise ValueError(f"Unsupported analysis policy: {policy!r}.")
    if dataset_transfer_mode not in {"upload", "reference"}:
        raise ValueError("dataset_transfer_mode must be 'upload' or 'reference'.")
    findings: list[CapabilityAssessment] = []

    def add(
        capability_id: str,
        path: str,
        category: str,
        capability: str,
        *,
        support: str = "supported",
        fidelity: str = "equivalent",
        action: str = "translate",
        verification: str = "unit",
        message: str,
        remediation: str | None = None,
        blocking: bool = False,
        external: bool = False,
        referenceable: bool | None = None,
        source_value: Any = None,
        target_value: Any = None,
    ) -> None:
        family = capability_family_for_id(capability_id)
        findings.append(
            CapabilityAssessment(
                capability_id=capability_id,
                family=family,
                path=path,
                category=category,
                capability=capability,
                support=support,
                semantic_fidelity=fidelity,
                selected_action=action,
                verification=verification,
                message=message,
                remediation=remediation,
                blocking=blocking,
                external_dependency=external,
                referenceable=referenceable,
                source_value=source_value,
                target_value=target_value,
            )
        )

    job_type = str(aml_job.get("type") or "command").lower()
    add(
        "job.type",
        "type",
        "job",
        "Command job",
        support="supported" if job_type == "command" else "unsupported",
        verification="live" if job_type == "command" else "none",
        message=(
            "Standalone command jobs are supported."
            if job_type == "command"
            else f"Job type {job_type!r} is outside this command-job migrator."
        ),
        remediation=(
            None
            if job_type == "command"
            else "Use a pipeline/sweep/AutoML-specific migration workflow."
        ),
        blocking=job_type != "command",
        source_value=job_type,
        target_value="Command" if job_type == "command" else None,
    )
    command = str(aml_job.get("command") or "")
    add(
        "job.command",
        "command",
        "job",
        "Command line",
        support="supported" if command else "unsupported",
        verification="live" if command else "none",
        message="The command is preserved with path-placeholder adaptations."
        if command
        else "No command was found.",
        blocking=not bool(command),
    )

    code = aml_job.get("code")
    if code or source_code_path:
        add(
            "asset.code",
            "code",
            "asset",
            "Code snapshot",
            action="copy",
            verification="live",
            message="Code is copied into a project Dataset V3 asset for codeId.",
            remediation=(
                "Code references are a future Dataset V3-reference candidate, "
                "but are not implemented or live-verified today."
            ),
            external=True,
            referenceable=False,
            source_value=str(source_code_path or code),
            target_value="Dataset V3 codeId",
        )
    else:
        add(
            "asset.code",
            "code",
            "asset",
            "Code snapshot",
            action="none",
            verification="unit",
            message="The job has no separate code snapshot.",
            referenceable=True,
        )

    inspected_environment = environment or EnvironmentInspection(
        source_reference=str(aml_job.get("environment") or "") or None,
        image_reference=str(aml_job.get("environment") or "") or None,
    )
    if inspected_environment.issue:
        add(
            "asset.environment",
            "environment",
            "environment",
            "Runtime environment",
            support="manual_action",
            fidelity="unknown",
            action="build",
            verification="none",
            message=inspected_environment.issue,
            remediation="Build and publish a complete OCI image, then pass --environment-image.",
            blocking=True,
            external=True,
            referenceable=False,
            source_value=inspected_environment.source_reference,
        )
    elif inspected_environment.image_reference:
        image = inspected_environment.image_reference
        mutable = "@sha256:" not in image.lower()
        private_acr = ".azurecr.io/" in image.lower()
        add(
            "asset.environment",
            "environment",
            "environment",
            "Runtime environment image",
            support="conditional" if mutable or private_acr else "supported",
            fidelity="unknown" if mutable else "equivalent",
            action="reference",
            verification="live",
            message="The OCI image is reused in place through environmentImageReference.",
            remediation=(
                "Pin the image by digest and ensure the target identity has "
                "AcrPull plus registry network/DNS access."
                if mutable or private_acr
                else None
            ),
            external=True,
            referenceable=True,
            source_value=inspected_environment.source_reference,
            target_value=image,
        )
    else:
        add(
            "asset.environment",
            "environment",
            "environment",
            "Runtime environment",
            support="unsupported",
            fidelity="unknown",
            action="none",
            verification="none",
            message="No reusable image reference was resolved.",
            remediation="Pass --environment-image with a runnable OCI image.",
            blocking=True,
            external=True,
            referenceable=False,
        )

    add(
        "job.compute",
        "compute",
        "compute",
        "Compute target",
        fidelity="adapted",
        action="replace",
        verification="live",
        message="The AML compute target and VM size are replaced by explicit Foundry compute settings.",
        source_value=aml_job.get("compute"),
    )
    source_identity = aml_job.get("identity")
    if source_identity and not user_assigned_identity_id:
        add(
            "job.identity",
            "identity",
            "identity",
            "Runtime identity",
            support="manual_action",
            fidelity="unknown",
            action="replace",
            verification="none",
            message="AML identity does not cross resource boundaries and no target UAI was supplied.",
            remediation="Supply --user-assigned-identity-id and grant its data, ACR, and secret permissions.",
            blocking=True,
            source_value=source_identity,
        )
    else:
        add(
            "job.identity",
            "identity",
            "identity",
            "Runtime identity",
            fidelity="adapted",
            action="replace" if source_identity else "configure",
            verification="live",
            message="The migrated job uses the explicitly configured target UAI.",
            source_value=source_identity,
            target_value=user_assigned_identity_id,
        )

    if permission_inspection is not None:
        identity_resolved = bool(permission_inspection.principal_id)
        add(
            "rbac.identity",
            "permissions.identity",
            "permission",
            "Target UAI principal resolution",
            support="supported" if identity_resolved else "manual_action",
            fidelity="equivalent" if identity_resolved else "unknown",
            action="verify" if identity_resolved else "resolve",
            verification="read_only",
            message=(
                "The target UAI exists and exposes a principal ID."
                if identity_resolved
                else "The target UAI principal could not be resolved."
            ),
            remediation=(
                None
                if identity_resolved
                else "Verify the UAI resource ID and grant the analyzer read access."
            ),
            blocking=not identity_resolved,
            source_value="satisfied" if identity_resolved else "unknown",
            target_value=permission_inspection.principal_id,
        )
        for check in permission_inspection.checks:
            satisfied = check.status == "satisfied"
            conditional = check.status == "conditional"
            support = (
                "supported"
                if satisfied
                else "conditional"
                if conditional
                else "manual_action"
            )
            remediation = None
            if check.status == "missing":
                remediation = (
                    "Attach --user-assigned-identity-id to the Foundry project "
                    "before submitting the job."
                    if check.requirement_id == "rbac.project_identity"
                    else (
                        "Assign one of "
                        + ", ".join(repr(role) for role in check.accepted_role_names)
                        + f" at {check.scope!r} or an ancestor scope."
                    )
                )
            elif check.status == "unknown":
                remediation = (
                    "Grant the analyzer Microsoft.Authorization/roleAssignments/read "
                    "and managed-identity read access, then rerun analysis."
                )
            elif conditional:
                remediation = (
                    "Evaluate the Azure RBAC condition against the exact storage, "
                    "repository, or project operation."
                )
            add(
                check.requirement_id,
                f"permissions.{check.requirement_id}",
                "permission",
                check.capability,
                support=support,
                fidelity="equivalent" if satisfied else "unknown",
                action="verify" if satisfied or conditional else "authorize",
                verification="read_only",
                message=check.message,
                remediation=remediation,
                blocking=check.blocking,
                source_value=check.status,
                target_value={
                    "principalId": check.principal_id,
                    "scope": check.scope,
                    "acceptedRoleNames": list(check.accepted_role_names),
                    "observedAssignments": [
                        assignment.to_dict()
                        for assignment in check.observed_assignments
                    ],
                    "error": check.error,
                },
            )

    if source_storage_connection_name:
        connection_blocking = source_storage_connection_available is False
        add(
            "connection.source_storage",
            "target.sourceStorageConnection",
            "connection",
            "Source storage connection",
            support="supported"
            if source_storage_connection_available
            else "manual_action",
            fidelity="equivalent" if source_storage_connection_available else "unknown",
            action="reference",
            verification="live" if source_storage_connection_available else "none",
            message=(
                "The Foundry source-storage connection exists."
                if source_storage_connection_available
                else "The configured Foundry source-storage connection could not be verified."
            ),
            remediation=(
                None
                if source_storage_connection_available
                else f"Create or fix connection {source_storage_connection_name!r}: {source_storage_connection_detail}"
            ),
            blocking=connection_blocking and dataset_transfer_mode == "reference",
            source_value=source_storage_connection_name,
            target_value=source_storage_connection_detail,
        )

    reference_uris = dict(data_reference_uris or {})
    reference_errors = dict(data_reference_errors or {})
    inputs = aml_job.get("inputs") or {}
    if isinstance(inputs, Mapping):
        for raw_name, raw_binding in inputs.items():
            name = str(raw_name)
            binding = (
                dict(raw_binding)
                if isinstance(raw_binding, Mapping)
                else {"value": raw_binding, "type": "literal"}
            )
            input_type = str(
                binding.get("type")
                or ("literal" if "value" in binding else "uri_folder")
            ).lower()
            path = f"inputs.{name}"
            if input_type in _LITERAL_TYPES:
                add(
                    f"input.{name}",
                    path,
                    "input",
                    f"Literal input {name}",
                    action="inline",
                    verification="live",
                    message="The concrete literal value is preserved.",
                    source_value=binding.get("value"),
                )
                constraints = [
                    key
                    for key in ("default", "optional", "min", "max", "enum")
                    if binding.get(key) is not None and binding.get(key) is not False
                ]
                if constraints:
                    add(
                        f"input.{name}.definition",
                        path,
                        "input",
                        f"Input definition metadata for {name}",
                        support="manual_action",
                        fidelity="lossy",
                        action="drop",
                        verification="none",
                        message=f"Definition fields are not represented: {', '.join(constraints)}.",
                        remediation="Keep the concrete value or recreate reusable parameter validation outside the job.",
                    )
                continue
            if input_type not in _DATA_TYPES | _MODEL_TYPES:
                add(
                    f"input.{name}",
                    path,
                    "input",
                    f"Input {name}",
                    support="unsupported",
                    fidelity="unknown",
                    action="none",
                    verification="none",
                    message=f"Input type {input_type!r} is unsupported.",
                    blocking=True,
                    source_value=binding,
                )
                continue
            mode = str(binding.get("mode") or "ro_mount").lower()
            if mode not in _INPUT_MODES:
                add(
                    f"input.{name}.mode",
                    f"{path}.mode",
                    "input",
                    f"Input delivery mode for {name}",
                    support="unsupported",
                    fidelity="unknown",
                    action="none",
                    verification="none",
                    message=f"Input mode {mode!r} is unsupported.",
                    blocking=True,
                    source_value=mode,
                )
            elif mode != "download":
                add(
                    f"input.{name}.mode",
                    f"{path}.mode",
                    "input",
                    f"Input delivery mode for {name}",
                    support="manual_action",
                    fidelity="unknown",
                    action="translate",
                    verification="unit",
                    message=(
                        f"Input mode {mode!r} has wire-level translation but no "
                        "completed AML-to-Foundry parity evidence."
                    ),
                    remediation=(
                        "Use download mode or run the extended migration "
                        "qualification before enabling this mode."
                    ),
                    blocking=True,
                    source_value=mode,
                    target_value=_INPUT_MODE_TARGETS[mode],
                )
            if input_type in _MODEL_TYPES:
                verified = "live" if input_type == "custom_model" else "unit"
                add(
                    f"input.{name}",
                    path,
                    "model",
                    f"{input_type} input {name}",
                    support="supported"
                    if input_type == "custom_model"
                    else "conditional",
                    fidelity="equivalent"
                    if input_type == "custom_model"
                    else "unknown",
                    action="copy",
                    verification=verified,
                    message="The model is downloaded and registered through Model V3.",
                    remediation=(
                        None
                        if input_type == "custom_model"
                        else "Run a format-specific parity test before production migration."
                    ),
                    blocking=input_type != "custom_model",
                    external=True,
                    referenceable=False,
                    source_value=binding.get("path") or binding.get("uri"),
                    target_value="Model V3",
                )
                continue
            resolved_uri = reference_uris.get(name)
            resolution_error = reference_errors.get(name)
            resolved_host = (
                urlsplit(resolved_uri).netloc.lower() if resolved_uri else ""
            )
            connection_host = (
                urlsplit(source_storage_connection_detail).netloc.lower()
                if source_storage_connection_detail
                and "://" in source_storage_connection_detail
                else ""
            )
            connection_matches = bool(
                source_storage_connection_available
                and resolved_host
                and connection_host
                and resolved_host == connection_host
            )
            can_reference = bool(resolved_uri and connection_matches)
            selected_action = (
                "reference" if dataset_transfer_mode == "reference" else "copy"
            )
            blocking = dataset_transfer_mode == "reference" and not can_reference
            support = (
                "manual_action"
                if blocking
                else ("conditional" if input_type == "mltable" else "supported")
            )
            fidelity = "unknown" if input_type == "mltable" else "equivalent"
            add(
                f"input.{name}",
                path,
                "data",
                f"{input_type} input {name}",
                support=support,
                fidelity=fidelity,
                action=selected_action,
                verification="live",
                message=(
                    "Dataset bytes are registered by reference without copying."
                    if selected_action == "reference" and can_reference
                    else "Dataset bytes are exported and copied into Dataset V3."
                    if selected_action == "copy"
                    else "The source data cannot yet be registered by reference."
                ),
                remediation=(
                    None
                    if not blocking
                    else (
                        "Provide a source-storage connection whose target host "
                        f"matches {resolved_host or 'the resolved data URI'}. "
                        f"Observed connection host: {connection_host or 'none'}. "
                        f"{resolution_error or ''}"
                    ).strip()
                ),
                blocking=blocking,
                external=True,
                referenceable=can_reference,
                source_value=binding.get("path") or binding.get("uri"),
                target_value=resolved_uri or "Dataset V3",
            )
            metadata = [
                key
                for key in ("path_on_compute", "datastore", "intellectual_property")
                if binding.get(key)
            ]
            if metadata:
                add(
                    f"input.{name}.metadata",
                    path,
                    "input",
                    f"Input metadata for {name}",
                    support="manual_action",
                    fidelity="lossy",
                    action="drop",
                    verification="none",
                    message=f"Metadata fields are not represented: {', '.join(metadata)}.",
                )

    outputs = aml_job.get("outputs") or {}
    if isinstance(outputs, Mapping):
        for raw_name, raw_binding in outputs.items():
            name = str(raw_name)
            binding = dict(raw_binding) if isinstance(raw_binding, Mapping) else {}
            output_type = str(binding.get("type") or "uri_folder").lower()
            mode = str(binding.get("mode") or "rw_mount").lower()
            path = f"outputs.{name}"
            source_output_path = str(binding.get("path") or binding.get("uri") or "")
            generated_default_output = (
                name == "default"
                and "workspaceartifactstore/experimentrun/"
                in source_output_path.lower()
            )
            if output_type not in _DATA_TYPES | _MODEL_TYPES:
                add(
                    f"output.{name}",
                    path,
                    "output",
                    f"Output {name}",
                    support="unsupported",
                    fidelity="unknown",
                    action="none",
                    verification="none",
                    message=f"Output type {output_type!r} is unsupported.",
                    blocking=True,
                    source_value=binding,
                )
                continue
            if mode not in _OUTPUT_MODES:
                add(
                    f"output.{name}.mode",
                    f"{path}.mode",
                    "output",
                    f"Output delivery mode for {name}",
                    support="unsupported",
                    fidelity="unknown",
                    action="none",
                    verification="none",
                    message=f"Output mode {mode!r} is unsupported.",
                    blocking=True,
                )
            elif mode != "upload" and not generated_default_output:
                add(
                    f"output.{name}.mode",
                    f"{path}.mode",
                    "output",
                    f"Output delivery mode for {name}",
                    support="manual_action",
                    fidelity="unknown",
                    action="translate",
                    verification="unit",
                    message=(
                        f"Output mode {mode!r} has wire-level translation but no "
                        "completed AML-to-Foundry parity evidence."
                    ),
                    remediation=(
                        "Use upload mode or run the extended migration "
                        "qualification before enabling this mode."
                    ),
                    blocking=True,
                    source_value=mode,
                    target_value=_OUTPUT_MODE_TARGETS[mode],
                )
            live_type = output_type in {
                "uri_file",
                "uri_folder",
                "mltable",
                "custom_model",
            }
            target_output_type = (
                "uri_folder" if output_type == "mltable" else output_type
            )
            add(
                f"output.{name}",
                path,
                "output",
                f"{output_type} output {name}",
                support="supported" if live_type else "conditional",
                fidelity=(
                    "lossy"
                    if source_output_path
                    else "adapted"
                    if live_type or output_type == "mltable"
                    else "unknown"
                ),
                action="transform" if output_type == "mltable" else "allocate",
                verification="live" if live_type else "unit",
                message=(
                    "Foundry registers the MLTable folder as a URI-folder asset; "
                    "the MLTable definition and payload files remain in that folder."
                    if output_type == "mltable"
                    else "Foundry allocates and registers a project-scoped output asset; the AML destination path is replaced."
                ),
                remediation=(
                    None
                    if live_type
                    else "Run an output-format-specific parity test before production migration."
                ),
                blocking=not live_type,
                source_value=source_output_path,
                target_value=f"project-scoped {target_output_type} output asset",
            )
            if binding.get("early_available"):
                add(
                    f"output.{name}.early_available",
                    f"{path}.early_available",
                    "output",
                    f"Early availability for {name}",
                    support="manual_action",
                    fidelity="lossy",
                    action="drop",
                    verification="none",
                    message="Early-available output semantics are not represented.",
                )

    env_vars = aml_job.get("environment_variables") or {}
    templated = (
        [str(name) for name, value in dict(env_vars).items() if "${{" in str(value)]
        if isinstance(env_vars, Mapping)
        else []
    )
    add(
        "job.environment_variables",
        "environment_variables",
        "environment",
        "Environment variables",
        fidelity="adapted" if templated else "equivalent",
        action="transform" if templated else "translate",
        verification="live",
        message=(
            f"Templated variables are moved into the command: {', '.join(templated)}."
            if templated
            else "Static environment variables are preserved."
        ),
    )

    resources = aml_job.get("resources") or {}
    instance_count = (
        int(resources.get("instance_count") or 1)
        if isinstance(resources, Mapping)
        else 1
    )
    add(
        "job.resources",
        "resources",
        "compute",
        "Resource configuration",
        fidelity="adapted",
        action="translate",
        verification="live" if instance_count == 1 else "unit",
        message=f"Instance count {instance_count} and shared memory are translated; the instance type is replaced.",
    )
    if isinstance(resources, Mapping) and (
        resources.get("docker_args") or resources.get("locations")
    ):
        add(
            "job.resources.aml_only",
            "resources",
            "compute",
            "AML-only resource settings",
            support="manual_action",
            fidelity="lossy",
            action="drop",
            verification="none",
            message="AML Docker arguments and location hints are not copied.",
            remediation="Bake required Docker behavior into the image and select an explicit Foundry compute target.",
        )

    distribution = aml_job.get("distribution")
    if isinstance(distribution, Mapping) and distribution:
        distribution_type = str(distribution.get("type") or "").lower()
        add(
            "job.distribution",
            "distribution",
            "compute",
            "Distributed execution",
            support="supported"
            if distribution_type in _DISTRIBUTIONS
            else "unsupported",
            fidelity="equivalent" if distribution_type in _DISTRIBUTIONS else "unknown",
            action="translate" if distribution_type in _DISTRIBUTIONS else "none",
            verification="unit" if distribution_type in _DISTRIBUTIONS else "none",
            message=(
                f"{distribution_type} distribution fields are translated."
                if distribution_type in _DISTRIBUTIONS
                else f"Distribution type {distribution_type!r} is unsupported."
            ),
            remediation="Run a migration-specific multi-node parity test before production use."
            if distribution_type in _DISTRIBUTIONS
            else None,
            blocking=distribution_type not in _DISTRIBUTIONS,
        )
    limits = aml_job.get("limits")
    if isinstance(limits, Mapping) and limits.get("timeout") is not None:
        add(
            "job.limits.timeout",
            "limits.timeout",
            "runtime",
            "Timeout",
            action="translate",
            verification="live",
            message="Timeout seconds are translated to an ISO-8601 Foundry command limit.",
        )
    services = aml_job.get("services") or {}
    if isinstance(services, Mapping):
        for raw_name, raw_service in services.items():
            name = str(raw_name)
            service = dict(raw_service) if isinstance(raw_service, Mapping) else {}
            service_type = str(service.get("type") or name).replace("_", "").lower()
            if service_type in _GENERATED_SERVICES:
                add(
                    f"service.{name}",
                    f"services.{name}",
                    "service",
                    f"Generated service {name}",
                    fidelity="adapted",
                    action="replace",
                    verification="live",
                    message="AML-generated Studio/Tracking links are replaced by Foundry-generated links.",
                )
            else:
                supported = service_type in _SUPPORTED_SERVICES
                add(
                    f"service.{name}",
                    f"services.{name}",
                    "service",
                    f"Interactive service {name}",
                    support="manual_action" if supported else "unsupported",
                    fidelity="unknown",
                    action="translate" if supported else "none",
                    verification="unit" if supported else "none",
                    message=(
                        f"Service type {service_type!r} is translated but has no "
                        "migration-specific endpoint-connectivity evidence."
                        if supported
                        else f"Service type {service_type!r} is unsupported."
                    ),
                    remediation=(
                        "Run the interactive migration qualification and provide "
                        "any required sidecar image, UAI, and SSH key settings."
                        if supported
                        else None
                    ),
                    blocking=True,
                )
    if (
        aml_job.get("queue_settings")
        or aml_job.get("job_tier")
        or aml_job.get("priority")
    ):
        add(
            "job.scheduling",
            "queue_settings",
            "scheduling",
            "Queue tier and priority",
            fidelity="adapted",
            action="translate",
            verification="unit",
            message="Source scheduling metadata is translated separately from the target AISuperComputer SLA tier.",
        )
    properties = aml_job.get("properties") or {}
    if isinstance(properties, Mapping) and properties:
        add(
            "job.properties",
            "properties",
            "metadata",
            "Custom job properties",
            fidelity="adapted",
            action="filter",
            verification="unit",
            message="Portable custom/MLflow properties are preserved; AML runtime-only properties are filtered.",
        )
    if (
        aml_job.get("parent_job_name")
        or aml_job.get("notification_setting")
        or aml_job.get("intellectual_property")
        or aml_job.get("parameters")
    ):
        add(
            "job.control_plane_metadata",
            "metadata",
            "metadata",
            "AML control-plane metadata",
            support="manual_action",
            fidelity="lossy",
            action="drop",
            verification="none",
            message="Parent lineage, notifications, IP metadata, and parameter metadata are not recreated.",
        )
    if aml_job.get("is_deterministic") is False:
        add(
            "job.is_deterministic",
            "is_deterministic",
            "runtime",
            "AML deterministic/reuse semantics",
            support="manual_action",
            fidelity="lossy",
            action="drop",
            verification="none",
            message="AML is_deterministic=False reuse/caching semantics are not represented.",
        )

    compatibility_warnings = audit_aml_command_job_compatibility(
        aml_job,
        user_assigned_identity_id=user_assigned_identity_id,
    )
    return MigrationAnalysisReport(
        source_job_name=str(aml_job.get("name") or ""),
        source_job_type=job_type,
        policy=policy,
        dataset_transfer_mode=dataset_transfer_mode,
        environment=inspected_environment,
        capabilities=tuple(findings),
        compatibility_warnings=compatibility_warnings,
        permission_inspection=permission_inspection,
    )


def analyze_aml_command_job(
    *,
    source: AmlWorkspace,
    target: FoundryTarget,
    source_job_name: str,
    policy: str = "advisory",
    dataset_transfer_mode: str = "upload",
    source_storage_connection_name: str | None = None,
    environment_image_reference: str | None = None,
    source_code_path: str | Path | None = None,
    check_permissions: bool = True,
    credential: Any | None = None,
    ml_client: Any | None = None,
) -> MigrationAnalysisReport:
    """Read AML/Foundry metadata and produce a no-side-effect analysis report."""

    if credential is None:
        from azure.identity import AzureCliCredential

        credential = AzureCliCredential(process_timeout=60)
    if ml_client is None:
        from azure.ai.ml import MLClient

        ml_client = MLClient(
            credential,
            source.subscription_id,
            source.resource_group,
            source.workspace_name,
        )
    source_entity = ml_client.jobs.get(source_job_name)
    materialized = materialize_aml_command_job(source_entity)
    environment = _inspect_environment(
        source_entity,
        source=source,
        credential=credential,
        ml_client=ml_client,
        override=environment_image_reference,
    )
    source_connection = (
        inspect_connection_permission_info(
            project_endpoint=target.project_endpoint,
            project_name=target.project_name,
            connection_name=source_storage_connection_name,
            credential=credential,
        )
        if source_storage_connection_name
        else ConnectionPermissionInfo(
            name="",
            available=False,
            error="No Foundry source-storage connection was supplied.",
        )
    )
    connection_available = (
        source_connection.available if source_storage_connection_name else None
    )
    connection_detail = source_connection.target or source_connection.error
    reference_uris: dict[str, str | None] = {}
    reference_errors: dict[str, str | None] = {}
    inputs = materialized.get("inputs") or {}
    if isinstance(inputs, Mapping):
        for raw_name, raw_binding in inputs.items():
            if not isinstance(raw_binding, Mapping):
                continue
            input_type = str(raw_binding.get("type") or "").lower()
            if input_type not in _DATA_TYPES:
                continue
            source_uri = str(raw_binding.get("path") or raw_binding.get("uri") or "")
            resolved, error = _resolve_data_reference_uri(
                source_uri,
                source=source,
                credential=credential,
                ml_client=ml_client,
            )
            reference_uris[str(raw_name)] = resolved
            reference_errors[str(raw_name)] = error
    permission_inspection = None
    if check_permissions:
        target_connection = inspect_connection_permission_info(
            project_endpoint=target.project_endpoint,
            project_name=target.project_name,
            connection_name=target.storage_connection_name,
            credential=credential,
        )
        subscription_ids = tuple(
            item
            for item in dict.fromkeys(
                (
                    source.subscription_id,
                    subscription_id_from_resource_id(target.user_assigned_identity_id),
                    subscription_id_from_resource_id(target.compute_id),
                )
            )
            if item
        )
        unresolved_checks: list[PermissionCheck] = []
        project_id = target_connection.project_id or source_connection.project_id
        if not project_id:
            unresolved_checks.append(
                PermissionCheck(
                    requirement_id="rbac.foundry_project",
                    capability="Foundry project runtime access",
                    status="unknown",
                    scope=None,
                    principal_id=None,
                    accepted_role_names=(
                        "Foundry User",
                        "Azure AI Developer",
                        "Azure AI Administrator",
                    ),
                    reason="The job identity must access the target Foundry project.",
                    message="The Foundry project ARM scope could not be derived from connection metadata.",
                    error=target_connection.error or source_connection.error,
                )
            )

        def connection_uses_aad(connection: ConnectionPermissionInfo) -> bool:
            credential_type = str(connection.credential_type or "").lower()
            return not credential_type or credential_type in {
                "aad",
                "managedidentity",
                "projectmanagedidentity",
            }

        target_storage_id = (
            target_connection.resource_id
            if target_connection.available and connection_uses_aad(target_connection)
            else None
        )
        if connection_uses_aad(target_connection) and not target_storage_id:
            unresolved_checks.append(
                PermissionCheck(
                    requirement_id="rbac.target_storage",
                    capability="Foundry output storage data-plane access",
                    status="unknown",
                    scope=None,
                    principal_id=None,
                    accepted_role_names=(
                        "Storage Blob Data Contributor",
                        "Storage Blob Data Owner",
                    ),
                    reason="The job identity must write Foundry outputs.",
                    message="The target storage ARM scope could not be derived from connection metadata.",
                    error=target_connection.error,
                )
            )

        source_storage_ids: list[str] = []
        if dataset_transfer_mode == "reference" and connection_uses_aad(
            source_connection
        ):
            for resolved_uri in sorted({uri for uri in reference_uris.values() if uri}):
                account_name = storage_account_name_from_uri(resolved_uri)
                if not account_name:
                    unresolved_checks.append(
                        PermissionCheck(
                            requirement_id="rbac.source_storage.unknown",
                            capability="Source storage read access",
                            status="unknown",
                            scope=None,
                            principal_id=None,
                            accepted_role_names=(
                                "Storage Blob Data Reader",
                                "Storage Blob Data Contributor",
                                "Storage Blob Data Owner",
                            ),
                            reason="Zero-copy inputs require source blob read access.",
                            message=f"No Azure Storage account could be derived from {resolved_uri!r}.",
                        )
                    )
                    continue
                resource_id = None
                if (
                    source_connection.resource_id
                    and urlsplit(source_connection.target or "").hostname
                    == urlsplit(resolved_uri).hostname
                ):
                    resource_id = source_connection.resource_id
                if resource_id is None:
                    try:
                        resource_id = resolve_storage_account_resource_id(
                            account_name,
                            subscription_ids,
                            credential=credential,
                        )
                    except Exception as error:
                        unresolved_checks.append(
                            PermissionCheck(
                                requirement_id=f"rbac.source_storage.{account_name}",
                                capability=f"Source storage read access ({account_name})",
                                status="unknown",
                                scope=None,
                                principal_id=None,
                                accepted_role_names=(
                                    "Storage Blob Data Reader",
                                    "Storage Blob Data Contributor",
                                    "Storage Blob Data Owner",
                                ),
                                reason="Zero-copy inputs require source blob read access.",
                                message="The source storage ARM scope could not be resolved.",
                                error=f"{type(error).__name__}: {error}",
                            )
                        )
                if resource_id:
                    source_storage_ids.append(resource_id)
                elif not any(
                    item.requirement_id == f"rbac.source_storage.{account_name}"
                    for item in unresolved_checks
                ):
                    unresolved_checks.append(
                        PermissionCheck(
                            requirement_id=f"rbac.source_storage.{account_name}",
                            capability=f"Source storage read access ({account_name})",
                            status="unknown",
                            scope=None,
                            principal_id=None,
                            accepted_role_names=(
                                "Storage Blob Data Reader",
                                "Storage Blob Data Contributor",
                                "Storage Blob Data Owner",
                            ),
                            reason="Zero-copy inputs require source blob read access.",
                            message="The source storage account was not found in the candidate subscriptions.",
                        )
                    )

        acr_resource_id = None
        registry_host = acr_host_from_image(environment.image_reference)
        if registry_host:
            try:
                acr_resource_id = resolve_acr_resource_id(
                    registry_host,
                    subscription_ids,
                    credential=credential,
                )
            except Exception as error:
                unresolved_checks.append(
                    PermissionCheck(
                        requirement_id="rbac.environment_acr",
                        capability="Private ACR image pull",
                        status="unknown",
                        scope=None,
                        principal_id=None,
                        accepted_role_names=(
                            "AcrPull",
                            "AcrPush",
                            "Container Registry Repository Reader",
                            "Container Registry Repository Writer",
                            "Container Registry Repository Contributor",
                        ),
                        reason="The job identity must pull the private image.",
                        message="The private ACR ARM scope could not be resolved.",
                        error=f"{type(error).__name__}: {error}",
                    )
                )
            if acr_resource_id is None and not any(
                item.requirement_id == "rbac.environment_acr"
                for item in unresolved_checks
            ):
                unresolved_checks.append(
                    PermissionCheck(
                        requirement_id="rbac.environment_acr",
                        capability="Private ACR image pull",
                        status="unknown",
                        scope=None,
                        principal_id=None,
                        accepted_role_names=(
                            "AcrPull",
                            "AcrPush",
                            "Container Registry Repository Reader",
                            "Container Registry Repository Writer",
                            "Container Registry Repository Contributor",
                        ),
                        reason="The job identity must pull the private image.",
                        message=f"Registry {registry_host!r} was not found in the candidate subscriptions.",
                    )
                )

        requirements = build_runtime_permission_requirements(
            foundry_project_id=project_id,
            target_storage_resource_id=target_storage_id,
            source_storage_resource_ids=source_storage_ids,
            acr_resource_id=acr_resource_id,
        )
        permission_inspection = inspect_runtime_permissions(
            identity_resource_id=target.user_assigned_identity_id,
            requirements=requirements,
            credential=credential,
        )
        if project_id and target.user_assigned_identity_id:
            project_identity_check = inspect_project_identity_attachment(
                project_id,
                target.user_assigned_identity_id,
                principal_id=permission_inspection.principal_id,
                credential=credential,
            )
            permission_inspection = RuntimePermissionInspection(
                permission_inspection.identity_resource_id,
                permission_inspection.principal_id,
                permission_inspection.client_id,
                (*permission_inspection.checks, project_identity_check),
                permission_inspection.limitations,
            )
        if unresolved_checks:
            permission_inspection = RuntimePermissionInspection(
                permission_inspection.identity_resource_id,
                permission_inspection.principal_id,
                permission_inspection.client_id,
                (*permission_inspection.checks, *unresolved_checks),
                permission_inspection.limitations,
            )
    return analyze_materialized_aml_command_job(
        materialized,
        policy=policy,
        dataset_transfer_mode=dataset_transfer_mode,
        source_storage_connection_name=source_storage_connection_name,
        source_storage_connection_available=connection_available,
        source_storage_connection_detail=connection_detail,
        environment=environment,
        user_assigned_identity_id=target.user_assigned_identity_id,
        source_code_path=source_code_path,
        data_reference_uris=reference_uris,
        data_reference_errors=reference_errors,
        permission_inspection=permission_inspection,
    )
