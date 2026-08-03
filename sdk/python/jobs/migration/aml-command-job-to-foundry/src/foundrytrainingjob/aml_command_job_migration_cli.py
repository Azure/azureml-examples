"""Command-line interface for AML-to-Foundry command-job migration."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from azure.identity import AzureCliCredential

from . import constants as foundry_constants
from .aml_command_job_equivalence import (
    MigrationEquivalenceRequest,
    run_aml_foundry_equivalence_exercise,
)
from .aml_command_job_analysis import (
    ANALYSIS_POLICIES,
    analyze_aml_command_job,
)
from .aml_command_job_permissions import grant_missing_reference_storage_access
from .aml_command_job_migration import create_foundry_asset_version
from .aml_command_job_release_validation import (
    ReleaseValidationRequest,
    run_aml_foundry_release_validation,
)
from .aml_command_job_migrator import (
    AmlCommandJobMigrator,
    AmlWorkspace,
    FoundryTarget,
    MigrationRequest,
)
from .e2e.sanitization import sanitize_for_report


def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _add_source_arguments(
    parser: argparse.ArgumentParser,
    *,
    require_export_compute: bool = True,
) -> None:
    parser.add_argument(
        "--source-subscription",
        default=_env("AML_MIGRATION_SOURCE_SUBSCRIPTION")
        or foundry_constants.DEFAULT_SUBSCRIPTION_ID,
    )
    resource_group = _env("AML_MIGRATION_SOURCE_RESOURCE_GROUP")
    workspace = _env("AML_MIGRATION_SOURCE_WORKSPACE")
    compute = _env("AML_MIGRATION_SOURCE_COMPUTE")
    identity_datastore = (
        _env("AML_MIGRATION_SOURCE_IDENTITY_DATASTORE")
        or "foundrymigrationidentityblob"
    )
    parser.add_argument(
        "--source-resource-group",
        default=resource_group,
        required=resource_group is None,
    )
    parser.add_argument(
        "--source-workspace",
        default=workspace,
        required=workspace is None,
    )
    parser.add_argument(
        "--source-export-compute",
        default=compute or (None if require_export_compute else ""),
        required=require_export_compute and compute is None,
        help="AML compute used by the batched data export job.",
    )
    parser.add_argument(
        "--source-identity-datastore",
        default=identity_datastore,
        help=(
            "Credentialless AML datastore alias created over workspaceblobstore "
            "for identity-based input/output access."
        ),
    )


def _add_target_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project-endpoint",
        default=foundry_constants.DEFAULT_PROJECT_ENDPOINT,
    )
    parser.add_argument(
        "--project-name",
        default=foundry_constants.DEFAULT_PROJECT_NAME,
    )
    parser.add_argument(
        "--storage-connection",
        default=foundry_constants.DEFAULT_STORAGE_CONNECTION_NAME,
        help="Foundry project storage connection used by Dataset V3 uploads.",
    )
    parser.add_argument(
        "--dataset-transfer-mode",
        choices=("upload", "reference"),
        default=_env("AML_MIGRATION_DATASET_TRANSFER_MODE") or "upload",
        help=(
            "Upload copies data into project storage (default). Reference "
            "registers source storage URIs without copying bytes."
        ),
    )
    parser.add_argument(
        "--source-storage-connection",
        default=_env("AML_MIGRATION_SOURCE_STORAGE_CONNECTION"),
        help=(
            "Foundry project connection to the AML source storage account; "
            "required with --dataset-transfer-mode reference."
        ),
    )
    parser.add_argument(
        "--foundry-compute-id",
        default=foundry_constants.DEFAULT_COMPUTE_ID,
    )
    parser.add_argument(
        "--foundry-instance-type",
        default=foundry_constants.DEFAULT_INSTANCE_TYPE,
    )
    parser.add_argument(
        "--foundry-api-version",
        default=foundry_constants.DEFAULT_API_VERSION,
    )
    parser.add_argument(
        "--target-job-tier",
        choices=("Preserve", "Standard", "Premium"),
        default="Premium",
        help=(
            "Foundry AISuperComputer SLA tier. Defaults to Premium; Preserve "
            "omits the target override."
        ),
    )
    parser.add_argument(
        "--user-assigned-identity-id",
        default=_env("FOUNDRY_TRAININGJOB__IDENTITY_UAI"),
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--asset-version")
    parser.add_argument("--environment-image")
    parser.add_argument(
        "--export-environment-image",
        default="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
    )
    parser.add_argument("--poll-interval", type=float, default=15.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--debug", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aml-foundry-migrate",
        description=(
            "Migrate an AML command job and its code/data/model dependencies to "
            "Foundry compute. Authentication comes from the active az login."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze",
        help="Analyze migration support without creating assets or jobs.",
    )
    _add_source_arguments(analyze, require_export_compute=False)
    _add_target_arguments(analyze)
    analyze.add_argument("--source-job", required=True)
    analyze.add_argument("--source-code-path", type=Path)
    analyze.add_argument("--environment-image")
    analyze.add_argument(
        "--analysis-policy",
        choices=ANALYSIS_POLICIES,
        default="advisory",
        help=(
            "advisory always exits 0; migratable blocks unsupported concrete "
            "jobs; lossless blocks lossy/unknown semantics; reference-only "
            "requires every external dependency to be referenceable; strict "
            "also requires live verification. Policy failure exits 2."
        ),
    )
    analyze.add_argument(
        "--report-file",
        type=Path,
        help="Optional JSON report path. Azure remains read-only.",
    )
    analyze.add_argument("--debug", action="store_true")

    migrate = subparsers.add_parser(
        "migrate",
        help="Migrate an existing AML command job.",
    )
    _add_source_arguments(migrate)
    _add_target_arguments(migrate)
    _add_runtime_arguments(migrate)
    migrate.add_argument("--source-job", required=True)
    migrate.add_argument("--source-code-path", type=Path)
    migrate.add_argument("--target-job-name")
    migrate.add_argument("--target-job-name-prefix")
    migrate.add_argument("--output-asset-name-prefix")
    migrate.add_argument("--no-wait", action="store_true")
    migrate.add_argument(
        "--preflight-policy",
        choices=ANALYSIS_POLICIES[1:],
        help=(
            "Run read-only analysis before migration and create nothing when "
            "the selected policy fails. Policy failure exits 2."
        ),
    )
    migrate.add_argument(
        "--grant-reference-storage-access",
        action="store_true",
        help=(
            "With reference transfer, grant the target UAI Storage Blob Data "
            "Reader on analyzer-confirmed missing source storage accounts. "
            "Implies migratable preflight and never modifies other roles."
        ),
    )

    exercise = subparsers.add_parser(
        "exercise",
        help=(
            "Create a capability-rich AML job, run it, migrate it, run it in "
            "Foundry, and validate its outputs."
        ),
    )
    _add_source_arguments(exercise)
    _add_target_arguments(exercise)
    _add_runtime_arguments(exercise)
    exercise.add_argument("--fixture-job-name")
    exercise.add_argument("--existing-source-job")
    exercise.add_argument("--existing-source-asset-version")
    exercise.add_argument("--target-job-name")
    exercise.add_argument("--validation-timeout", type=float, default=600.0)

    qualify = subparsers.add_parser(
        "qualify-release",
        help=(
            "Run one rich AML fixture through upload and zero-copy migration, "
            "then save a fail-closed release evidence report."
        ),
    )
    _add_source_arguments(qualify)
    _add_target_arguments(qualify)
    _add_runtime_arguments(qualify)
    qualify.add_argument("--fixture-job-name")
    qualify.add_argument("--existing-source-job")
    qualify.add_argument("--existing-source-asset-version")
    qualify.add_argument(
        "--reuse-completed-upload",
        action="store_true",
        help=(
            "Revalidate work-dir/upload/equivalence-report.json and run only "
            "the missing reference migration."
        ),
    )
    qualify.add_argument("--validation-timeout", type=float, default=600.0)
    return parser


def _source(args: argparse.Namespace) -> AmlWorkspace:
    return AmlWorkspace(
        subscription_id=args.source_subscription,
        resource_group=args.source_resource_group,
        workspace_name=args.source_workspace,
        export_compute=args.source_export_compute,
        identity_datastore_name=args.source_identity_datastore,
    )


def _target(args: argparse.Namespace) -> FoundryTarget:
    return FoundryTarget(
        project_endpoint=args.project_endpoint,
        project_name=args.project_name,
        storage_connection_name=args.storage_connection,
        compute_id=args.foundry_compute_id,
        instance_type=args.foundry_instance_type,
        api_version=args.foundry_api_version,
        user_assigned_identity_id=args.user_assigned_identity_id,
        job_tier=None if args.target_job_tier == "Preserve" else args.target_job_tier,
    )


def _default_work_dir(label: str) -> Path:
    safe_label = (
        "".join(
            character if character.isalnum() or character in "_.-" else "-"
            for character in label
        ).strip("-.")
        or "run"
    )
    configured_root = _env("AML_MIGRATION_RUNS_DIR")
    runs_root = (
        Path(configured_root).expanduser()
        if configured_root
        else Path.home() / ".aml-foundry-migration" / "runs"
    )
    return runs_root / safe_label


def _migration_json(result: Any) -> dict[str, Any]:
    return {
        "manifestPath": result.manifest_path,
        "sourceJobName": result.source_job_name,
        "targetJobName": result.target_job_name,
        "targetStatus": result.target_status,
        "assetMappings": result.asset_mappings,
        "warnings": list(result.warnings),
    }


def _run_analyze(
    args: argparse.Namespace,
    credential: Any,
    *,
    policy: str | None = None,
) -> Any:
    return analyze_aml_command_job(
        source=_source(args),
        target=_target(args),
        source_job_name=args.source_job,
        policy=policy or args.analysis_policy,
        dataset_transfer_mode=args.dataset_transfer_mode,
        source_storage_connection_name=args.source_storage_connection,
        environment_image_reference=args.environment_image,
        source_code_path=args.source_code_path,
        credential=credential,
    )


def _run_migrate(args: argparse.Namespace, credential: Any) -> dict[str, Any]:
    work_dir = args.work_dir or _default_work_dir(args.source_job)
    request = MigrationRequest(
        source=_source(args),
        target=_target(args),
        source_job_name=args.source_job,
        work_dir=work_dir,
        environment_image_reference=args.environment_image,
        source_code_path=args.source_code_path,
        target_job_name=args.target_job_name,
        target_job_name_prefix=args.target_job_name_prefix,
        output_asset_name_prefix=args.output_asset_name_prefix,
        asset_version=args.asset_version,
        dataset_transfer_mode=args.dataset_transfer_mode,
        source_storage_connection_name=args.source_storage_connection,
        wait_for_completion=not args.no_wait,
        poll_interval_seconds=args.poll_interval,
        timeout_seconds=args.timeout,
        export_environment_image=args.export_environment_image,
    )
    result = AmlCommandJobMigrator(request, credential=credential).migrate()
    return {"migration": _migration_json(result)}


def _prepare_reference_storage_access(
    args: argparse.Namespace,
    credential: Any,
) -> tuple[Any, tuple[Any, ...]]:
    if args.dataset_transfer_mode != "reference":
        raise ValueError(
            "--grant-reference-storage-access requires "
            "--dataset-transfer-mode reference."
        )
    if not str(args.user_assigned_identity_id or "").strip():
        raise ValueError(
            "--grant-reference-storage-access requires " "--user-assigned-identity-id."
        )
    if args.preflight_policy not in (None, "migratable"):
        raise ValueError(
            "--grant-reference-storage-access supports only the migratable "
            "preflight policy; run stricter analysis separately before "
            "requesting an RBAC mutation."
        )

    analysis = _run_analyze(args, credential, policy="migratable")
    inspection = analysis.permission_inspection
    grantable_ids = (
        {
            check.requirement_id
            for check in inspection.checks
            if check.requirement_id.startswith("rbac.source_storage.")
            and check.status == "missing"
        }
        if inspection is not None
        else set()
    )
    non_grantable_blockers = {
        capability.capability_id
        for capability in analysis.capabilities
        if capability.blocking and capability.capability_id not in grantable_ids
    }
    if non_grantable_blockers or not grantable_ids:
        return analysis, ()

    grants = grant_missing_reference_storage_access(
        inspection,
        credential=credential,
    )
    if grants:
        work_dir = args.work_dir or _default_work_dir(args.source_job)
        evidence_path = work_dir / "reference-storage-role-assignments.json"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence = sanitize_for_report(
            {
                "schemaVersion": 1,
                "createdAtUtc": datetime.now(timezone.utc).isoformat(),
                "assignments": [grant.to_dict() for grant in grants],
            }
        )
        temporary_path = evidence_path.with_suffix(evidence_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(evidence_path)
    refreshed = _run_analyze(args, credential, policy="migratable")
    return refreshed, grants


def _run_exercise(args: argparse.Namespace, credential: Any) -> dict[str, Any]:
    version = args.asset_version or create_foundry_asset_version()
    label = args.fixture_job_name or f"exercise-{version}"
    work_dir = args.work_dir or _default_work_dir(label)
    source = _source(args)
    target = _target(args)
    result = run_aml_foundry_equivalence_exercise(
        MigrationEquivalenceRequest(
            source=source,
            target=target,
            work_dir=work_dir,
            environment_image=(
                args.environment_image
                or "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
            ),
            export_environment_image=args.export_environment_image,
            fixture_job_name=args.fixture_job_name,
            target_job_name=args.target_job_name,
            asset_version=version,
            dataset_transfer_mode=args.dataset_transfer_mode,
            source_storage_connection_name=args.source_storage_connection,
            poll_interval_seconds=args.poll_interval,
            timeout_seconds=args.timeout,
            validation_timeout_seconds=args.validation_timeout,
            existing_source_job_name=args.existing_source_job,
            existing_source_asset_version=args.existing_source_asset_version,
        ),
        credential=credential,
    )
    return result.to_dict()


def _run_qualify_release(
    args: argparse.Namespace,
    credential: Any,
) -> dict[str, Any]:
    version = args.asset_version or create_foundry_asset_version()
    label = args.fixture_job_name or f"release-{version}"
    work_dir = args.work_dir or _default_work_dir(label)
    result = run_aml_foundry_release_validation(
        ReleaseValidationRequest(
            source=_source(args),
            target=_target(args),
            source_storage_connection_name=args.source_storage_connection or "",
            work_dir=work_dir,
            environment_image=(
                args.environment_image
                or "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"
            ),
            export_environment_image=args.export_environment_image,
            fixture_job_name=args.fixture_job_name,
            asset_version=version,
            poll_interval_seconds=args.poll_interval,
            timeout_seconds=args.timeout,
            validation_timeout_seconds=args.validation_timeout,
            existing_source_job_name=args.existing_source_job,
            existing_source_asset_version=args.existing_source_asset_version,
            reuse_completed_upload=args.reuse_completed_upload,
        ),
        credential=credential,
    )
    return result.to_dict()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        credential = AzureCliCredential(process_timeout=60)
        analysis_report = None
        reference_storage_grants: tuple[Any, ...] = ()
        if args.command == "analyze":
            analysis_report = _run_analyze(args, credential)
            result = {"analysis": analysis_report.to_dict()}
        elif args.command == "migrate":
            if args.grant_reference_storage_access:
                (
                    analysis_report,
                    reference_storage_grants,
                ) = _prepare_reference_storage_access(args, credential)
            elif args.preflight_policy:
                analysis_report = _run_analyze(
                    args,
                    credential,
                    policy=args.preflight_policy,
                )
            if analysis_report is not None and not analysis_report.policy_passed:
                result = {"analysis": analysis_report.to_dict()}
            else:
                result = _run_migrate(args, credential)
                if analysis_report is not None:
                    result["analysis"] = analysis_report.to_dict()
            if args.grant_reference_storage_access:
                result["referenceStorageRoleAssignments"] = [
                    grant.to_dict() for grant in reference_storage_grants
                ]
        elif args.command == "exercise":
            result = _run_exercise(args, credential)
        else:
            result = _run_qualify_release(args, credential)
        result = sanitize_for_report(result)
        result["completedAtUtc"] = datetime.now(timezone.utc).isoformat()
        if args.command == "analyze" and args.report_file:
            report_path = args.report_file.expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = report_path.with_suffix(report_path.suffix + ".tmp")
            temporary_path.write_text(
                json.dumps(result, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary_path.replace(report_path)
        print(json.dumps(result, indent=2, sort_keys=True))
        if analysis_report is not None and not analysis_report.policy_passed:
            return 2
        return 0
    except Exception as error:
        if args.debug:
            raise
        print(
            f"aml-foundry-migrate failed: {sanitize_for_report(str(error))}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
