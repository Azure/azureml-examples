"""Translate Azure Machine Learning command-job definitions to Foundry jobs."""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


_INPUT_MODE_MAP = {
    "download": "Download",
    "direct": "Direct",
    "eval_download": "EvalDownload",
    "eval_mount": "EvalMount",
    "ro_mount": "ReadOnlyMount",
    "rw_mount": "ReadWriteMount",
}
_OUTPUT_MODE_MAP = {
    "direct": "Direct",
    "rw_mount": "ReadWriteMount",
    "upload": "ReadWriteMount",
}
_MODEL_TYPES = frozenset(
    {
        "custom_model",
        "mlflow_model",
        "triton_model",
    }
)
_LITERAL_TYPES = frozenset({"literal", "string", "integer", "number", "boolean"})
_SUPPORTED_INPUT_TYPES = frozenset(
    {*_LITERAL_TYPES, "uri_file", "uri_folder", "mltable", *_MODEL_TYPES}
)
_SUPPORTED_OUTPUT_TYPES = frozenset(
    {"uri_file", "uri_folder", "mltable", *_MODEL_TYPES}
)
_TEMPLATE_PATTERN = re.compile(r"\$\{\{\s*(?:inputs|outputs)\.[^{}]+\s*\}\}")
_DEFAULT_OUTPUT_TEMPLATE_PATTERN = re.compile(r"\$\{\{\s*outputs\.default\s*\}\}")
_ENVIRONMENT_VARIABLE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ASSET_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
_DISTRIBUTION_TYPES = {
    "mpi": "Mpi",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "ray": "Ray",
}
_GENERATED_SERVICE_TYPES = frozenset({"local", "studio", "tracking"})
_SERVICE_TYPES = {
    "jupyter": "Jupyter",
    "jupyterlab": "JupyterLab",
    "ssh": "SSH",
    "tensorboard": "TensorBoard",
    "vscode": "VSCode",
    "theia": "Theia",
    "grafana": "Grafana",
    "custom": "Custom",
}
_AML_RUNTIME_PROPERTY_NAMES = frozenset(
    {
        "contentsnapshotid",
        "processinfofile",
        "processstatusfile",
        "starttimeutc",
        "endtimeutc",
    }
)


@dataclass(frozen=True)
class TranslationResult:
    """Translated Foundry request plus non-fatal migration decisions."""

    request_body: dict[str, Any]
    warnings: tuple[str, ...]


def create_foundry_asset_version() -> str:
    """Create the 17-digit UTC version format used by Foundry job outputs."""

    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:17]


def _property(source: Mapping[str, Any], snake_name: str, camel_name: str) -> Any:
    if snake_name in source:
        return source[snake_name]
    return source.get(camel_name)


def _normalize_mode(value: Any, *, output: bool) -> str:
    mode_map = _OUTPUT_MODE_MAP if output else _INPUT_MODE_MAP
    default_mode = "ReadWriteMount" if output else "ReadOnlyMount"
    if value is None:
        return default_mode

    normalized = str(value).strip()
    if not normalized:
        return default_mode
    mapped = mode_map.get(normalized.lower())
    if mapped is not None:
        return mapped
    if normalized in mode_map.values():
        return normalized
    raise ValueError(f"Unsupported {'output' if output else 'input'} mode: {value!r}")


def _normalize_asset_name(value: str) -> str:
    normalized = _ASSET_NAME_PATTERN.sub("-", value).strip("-.")
    if not normalized:
        raise ValueError(f"Could not derive a Foundry asset name from {value!r}.")
    return normalized[:255]


def _looks_like_aml_asset_reference(value: str) -> bool:
    normalized = value.lower()
    return (
        normalized.startswith("azureml:")
        or normalized.startswith("/subscriptions/")
        or "/providers/microsoft.machinelearningservices/" in normalized
        or (
            "://" not in normalized
            and re.fullmatch(
                r"[^:@/]+(?::[^/?#]+|@[^/?#]+)",
                normalized,
            )
            is not None
        )
    )


def _migrated_asset_id(
    source_uri: str,
    *,
    migrated_asset_ids: Mapping[str, str],
    binding_name: str,
) -> str:
    migrated_uri = migrated_asset_ids.get(source_uri)
    if migrated_uri:
        return migrated_uri
    if _looks_like_aml_asset_reference(source_uri):
        raise ValueError(
            f"Input {binding_name!r} references AML asset {source_uri!r}, but no "
            "migrated Foundry asset ID was supplied."
        )
    return source_uri


def _portable_job_properties(value: Any) -> tuple[dict[str, str], tuple[str, ...]]:
    if not isinstance(value, Mapping):
        return {}, ()
    portable: dict[str, str] = {}
    dropped: list[str] = []
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        normalized = key.lower()
        if (
            normalized.startswith("_azureml.")
            or normalized.startswith("azureml.")
            or normalized in _AML_RUNTIME_PROPERTY_NAMES
        ):
            dropped.append(key)
            continue
        portable[key] = str(raw_value)
    return portable, tuple(sorted(dropped))


def audit_aml_command_job_compatibility(
    aml_job: Mapping[str, Any],
    *,
    user_assigned_identity_id: str | None = None,
) -> tuple[str, ...]:
    """Report source semantics that cannot be preserved by Foundry translation."""

    warnings: list[str] = []
    source_identity = aml_job.get("identity")
    if source_identity:
        target_identity = (
            repr(str(user_assigned_identity_id).rstrip("/"))
            if user_assigned_identity_id
            else "no explicit target identity"
        )
        warnings.append(
            f"Job identity {source_identity!r}: not copied across resource "
            f"boundaries; the migrated job uses {target_identity}."
        )
    if aml_job.get("parent_job_name"):
        warnings.append(
            f"Parent job {aml_job['parent_job_name']!r}: lineage is not copied; "
            "the migrated command job is submitted as a new root job."
        )
    if aml_job.get("is_deterministic") is False:
        warnings.append(
            "is_deterministic=False: AML reuse/caching semantics are not "
            "represented by Foundry command jobs."
        )
    _, dropped_property_names = _portable_job_properties(aml_job.get("properties"))
    if dropped_property_names:
        warnings.append(
            "AML runtime properties are not copied: "
            + ", ".join(repr(name) for name in dropped_property_names)
            + "."
        )
    for field_name, label in (
        ("notification_setting", "Job notification settings"),
        ("intellectual_property", "Job intellectual-property metadata"),
        ("parameters", "Job parameter metadata"),
    ):
        if aml_job.get(field_name):
            warnings.append(f"{label}: not copied to the Foundry job.")

    resources = aml_job.get("resources")
    if isinstance(resources, Mapping):
        source_instance_type = _property(resources, "instance_type", "instanceType")
        if source_instance_type:
            warnings.append(
                f"Source instance type {source_instance_type!r}: replaced by the "
                "explicit Foundry target instance type."
            )

    for collection_name in ("inputs", "outputs"):
        bindings = aml_job.get(collection_name)
        if not isinstance(bindings, Mapping):
            continue
        for raw_name, raw_binding in bindings.items():
            if not isinstance(raw_binding, Mapping):
                continue
            name = str(raw_name)
            binding = dict(raw_binding)
            unsupported_fields = (
                ("path_on_compute", "fixed compute path"),
                ("datastore", "source datastore selection"),
                ("intellectual_property", "intellectual-property metadata"),
            )
            if collection_name == "inputs":
                unsupported_fields += (
                    ("default", "default value metadata"),
                    ("optional", "optional-input semantics"),
                    ("min", "minimum-value constraint"),
                    ("max", "maximum-value constraint"),
                    ("enum", "enumerated-value constraint"),
                )
            else:
                unsupported_fields += (
                    ("early_available", "early-available output semantics"),
                )
            for field_name, label in unsupported_fields:
                value = binding.get(field_name)
                if value is not None and value is not False and value != ():
                    warnings.append(
                        f"{collection_name[:-1].capitalize()} {name!r} {label}: "
                        "not copied to Foundry."
                    )

            binding_type = str(
                _property(
                    binding,
                    "type",
                    "jobInputType" if collection_name == "inputs" else "jobOutputType",
                )
                or ""
            ).lower()
            if binding_type in _MODEL_TYPES and binding_type != "custom_model":
                warnings.append(
                    f"{collection_name[:-1].capitalize()} {name!r} uses "
                    f"{binding_type!r}; translation preserves the discriminator, "
                    "but only custom_model migration has live parity coverage."
                )
            if collection_name == "outputs" and binding_type == "mltable":
                warnings.append(
                    f"Output {name!r}: adapted AML 'mltable' to Foundry "
                    "'uri_folder'; Foundry output registration supports URI "
                    "folders/files, and the MLTable definition remains in the folder."
                )
    return tuple(dict.fromkeys(warnings))


def _environment_compatibility_warnings(
    image_reference: str,
    *,
    user_assigned_identity_id: str | None,
) -> tuple[str, ...]:
    normalized = str(image_reference).strip()
    warnings: list[str] = []
    if "@sha256:" not in normalized.lower():
        warnings.append(
            f"Environment image {normalized!r} is tag-based rather than "
            "digest-pinned; the migrated job can resolve different bytes later."
        )
    if ".azurecr.io/" in normalized.lower():
        identity_text = (
            f"target identity {str(user_assigned_identity_id).rstrip('/')!r}"
            if user_assigned_identity_id
            else "the target job identity"
        )
        warnings.append(
            f"Private ACR image {normalized!r} is reused in place, not copied; "
            f"ensure {identity_text} has AcrPull and network/DNS access to the registry."
        )
    return tuple(warnings)


def _translate_inputs(
    inputs: Any,
    *,
    migrated_asset_ids: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    if inputs is None:
        return {}
    if not isinstance(inputs, Mapping):
        raise ValueError("AML job inputs must be a mapping.")

    translated: dict[str, dict[str, Any]] = {}
    for raw_name, raw_binding in inputs.items():
        name = str(raw_name)
        if not isinstance(raw_binding, Mapping):
            translated[name] = {
                "jobInputType": "literal",
                "value": str(raw_binding),
            }
            continue

        binding = dict(raw_binding)
        input_type = str(
            _property(binding, "type", "jobInputType")
            or ("literal" if "value" in binding else "uri_folder")
        ).lower()
        if input_type not in _SUPPORTED_INPUT_TYPES:
            raise ValueError(f"Unsupported AML input type for {name!r}: {input_type!r}")

        if input_type in _LITERAL_TYPES:
            if "value" not in binding:
                raise ValueError(f"Literal AML input {name!r} has no value.")
            translated_binding: dict[str, Any] = {
                "jobInputType": "literal",
                "value": str(binding["value"]),
            }
        else:
            source_uri = _property(binding, "path", "uri")
            if not source_uri:
                raise ValueError(f"AML asset input {name!r} has no path/uri.")
            translated_binding = {
                "jobInputType": input_type,
                "uri": _migrated_asset_id(
                    str(source_uri),
                    migrated_asset_ids=migrated_asset_ids,
                    binding_name=name,
                ),
                "mode": _normalize_mode(binding.get("mode"), output=False),
            }

        description = binding.get("description")
        if description:
            translated_binding["description"] = str(description)
        translated[name] = translated_binding
    return translated


def _translate_outputs(
    outputs: Any,
    *,
    asset_name_prefix: str,
    asset_version: str,
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    if outputs is None:
        return {}
    if not isinstance(outputs, Mapping):
        raise ValueError("AML job outputs must be a mapping.")

    translated: dict[str, dict[str, Any]] = {}
    for raw_name, raw_binding in outputs.items():
        name = str(raw_name)
        binding = dict(raw_binding) if isinstance(raw_binding, Mapping) else {}
        output_type = str(
            _property(binding, "type", "jobOutputType") or "uri_folder"
        ).lower()
        if output_type not in _SUPPORTED_OUTPUT_TYPES:
            raise ValueError(
                f"Unsupported AML output type for {name!r}: {output_type!r}"
            )

        foundry_output_type = "uri_folder" if output_type == "mltable" else output_type
        translated_binding: dict[str, Any] = {
            "jobOutputType": foundry_output_type,
            "mode": _normalize_mode(binding.get("mode"), output=True),
            "assetName": _normalize_asset_name(f"{asset_name_prefix}-{name}"),
            "assetVersion": asset_version,
        }
        if output_type == "mltable":
            warnings.append(
                f"Output {name!r}: adapted AML 'mltable' to Foundry "
                "'uri_folder' because Foundry data output registration does not "
                "support the MLTable asset discriminator."
            )
        description = binding.get("description")
        if description:
            translated_binding["description"] = str(description)
        tags = binding.get("tags")
        if isinstance(tags, Mapping) and tags:
            translated_binding["tags"] = {
                str(key): str(value) for key, value in tags.items()
            }

        source_path = _property(binding, "path", "uri")
        if source_path:
            warnings.append(
                f"Output {name!r}: dropped AML datastore path {source_path!r}; "
                "Foundry will allocate and register a project-scoped output asset."
            )
        translated[name] = translated_binding
    return translated


def _prepare_default_output(
    command: str,
    outputs: Any,
    environment_variables: Any,
    *,
    warnings: list[str],
) -> tuple[str, Any, Any]:
    if not isinstance(outputs, Mapping) or "default" not in outputs:
        return command, outputs, environment_variables

    default_binding = outputs.get("default")
    binding = dict(default_binding) if isinstance(default_binding, Mapping) else {}
    source_path = str(_property(binding, "path", "uri") or "").lower()
    environment_values = (
        environment_variables.values()
        if isinstance(environment_variables, Mapping)
        else ()
    )
    is_referenced = bool(_DEFAULT_OUTPUT_TEMPLATE_PATTERN.search(command)) or any(
        _DEFAULT_OUTPUT_TEMPLATE_PATTERN.search(str(value))
        for value in environment_values
    )
    is_generated_artifact = (
        "workspaceartifactstore" in source_path
        and "/experimentrun/dcid." in source_path
    )
    prepared_outputs = deepcopy(dict(outputs))
    if is_generated_artifact and not is_referenced:
        prepared_outputs.pop("default", None)
        warnings.append(
            "Output 'default': skipped the generated AML run-artifact output; "
            "Foundry reserves this output name and generates its own job artifacts."
        )
        return command, prepared_outputs, environment_variables

    replacement_name = "aml_default"
    while replacement_name in prepared_outputs:
        replacement_name += "_output"
    prepared_outputs[replacement_name] = prepared_outputs.pop("default")
    replacement = f"${{{{outputs.{replacement_name}}}}}"
    command = _DEFAULT_OUTPUT_TEMPLATE_PATTERN.sub(replacement, command)
    if isinstance(environment_variables, Mapping):
        environment_variables = {
            key: _DEFAULT_OUTPUT_TEMPLATE_PATTERN.sub(replacement, str(value))
            for key, value in environment_variables.items()
        }
    warnings.append(
        f"Output 'default': renamed to {replacement_name!r} because Foundry "
        "reserves the default output name."
    )
    return command, prepared_outputs, environment_variables


def _prepare_file_output_paths(
    command: str,
    outputs: Any,
    environment_variables: Any,
    *,
    warnings: list[str],
) -> tuple[str, Any]:
    if not isinstance(outputs, Mapping):
        return command, environment_variables
    environment_mapping = (
        dict(environment_variables)
        if isinstance(environment_variables, Mapping)
        else environment_variables
    )
    for raw_name, raw_binding in outputs.items():
        name = str(raw_name)
        binding = dict(raw_binding) if isinstance(raw_binding, Mapping) else {}
        output_type = str(
            _property(binding, "type", "jobOutputType") or "uri_folder"
        ).lower()
        if output_type != "uri_file":
            continue
        source_path = str(_property(binding, "path", "uri") or "").rstrip("/")
        file_name = source_path.rsplit("/", 1)[-1] if "/" in source_path else ""
        if not file_name or ":" in file_name:
            file_name = name
        pattern = re.compile(r"\$\{\{\s*outputs\." + re.escape(name) + r"\s*\}\}")
        replacement = f"${{{{outputs.{name}}}}}/{file_name}"
        command = pattern.sub(replacement, command)
        if isinstance(environment_mapping, dict):
            environment_mapping = {
                key: pattern.sub(replacement, str(value))
                for key, value in environment_mapping.items()
            }
        warnings.append(
            f"Output {name!r}: appended file name {file_name!r} because Foundry "
            "resolves uri_file output placeholders to directories."
        )
    return command, environment_mapping


def _prepare_model_input_paths(
    command: str,
    environment_variables: Any,
    model_input_path_suffixes: Mapping[str, str],
    *,
    warnings: list[str],
) -> tuple[str, Any]:
    environment_mapping = (
        dict(environment_variables)
        if isinstance(environment_variables, Mapping)
        else environment_variables
    )
    for raw_name, raw_suffix in model_input_path_suffixes.items():
        name = str(raw_name)
        suffix = str(raw_suffix).strip("/")
        if not suffix:
            raise ValueError(f"Model input {name!r} has an empty path suffix.")
        pattern = re.compile(r"\$\{\{\s*inputs\." + re.escape(name) + r"\s*\}\}")
        replacement = f"${{{{inputs.{name}}}}}/{suffix}"
        command = pattern.sub(replacement, command)
        if isinstance(environment_mapping, dict):
            environment_mapping = {
                key: pattern.sub(replacement, str(value))
                for key, value in environment_mapping.items()
            }
        warnings.append(
            f"Input {name!r}: appended model payload path {suffix!r} because "
            "Foundry Model V3 exposes service metadata at the asset root."
        )
    return command, environment_mapping


def _prepare_input_path_suffixes(
    command: str,
    environment_variables: Any,
    input_path_suffixes: Mapping[str, str],
    *,
    warnings: list[str],
) -> tuple[str, Any]:
    environment_mapping = (
        dict(environment_variables)
        if isinstance(environment_variables, Mapping)
        else environment_variables
    )
    for raw_name, raw_suffix in input_path_suffixes.items():
        name = str(raw_name)
        suffix = str(raw_suffix).strip("/")
        if not suffix:
            raise ValueError(f"Input {name!r} has an empty path suffix.")
        pattern = re.compile(r"\$\{\{\s*inputs\." + re.escape(name) + r"\s*\}\}")
        replacement = f"${{{{inputs.{name}}}}}/{suffix}"
        command = pattern.sub(replacement, command)
        if isinstance(environment_mapping, dict):
            environment_mapping = {
                key: pattern.sub(replacement, str(value))
                for key, value in environment_mapping.items()
            }
        warnings.append(
            f"Input {name!r}: appended source-storage path {suffix!r} because "
            "Foundry Dataset V3 folder references bind the storage container root."
        )
    return command, environment_mapping


def _translate_templated_environment_variables(
    command: str,
    environment_variables: Any,
    *,
    warnings: list[str],
) -> tuple[str, dict[str, str]]:
    if environment_variables is None:
        return command, {}
    if not isinstance(environment_variables, Mapping):
        raise ValueError("AML environment_variables must be a mapping.")

    retained: dict[str, str] = {}
    shell_assignments: list[str] = []
    for raw_name, raw_value in environment_variables.items():
        name = str(raw_name)
        value = str(raw_value)
        if not _TEMPLATE_PATTERN.search(value):
            retained[name] = value
            continue
        if not _ENVIRONMENT_VARIABLE_PATTERN.fullmatch(name):
            raise ValueError(
                f"Cannot preserve templated AML environment variable {name!r}: "
                "the name is not shell-safe."
            )
        shell_assignments.append(f"export {name}={shlex.quote(value)}")
        warnings.append(
            f"Environment variable {name!r}: moved into the command because "
            "Foundry expands input/output placeholders only in properties.command."
        )

    if shell_assignments:
        command = f"{' && '.join(shell_assignments)} && {command}"
    return command, retained


def _copy_optional_mapping(
    source: Mapping[str, Any],
    snake_name: str,
    camel_name: str,
) -> dict[str, Any] | None:
    value = _property(source, snake_name, camel_name)
    return deepcopy(dict(value)) if isinstance(value, Mapping) and value else None


def _camelize_mapping(source: Mapping[str, Any]) -> dict[str, Any]:
    translated: dict[str, Any] = {}
    for key, value in source.items():
        parts = str(key).split("_")
        translated_key = parts[0] + "".join(part.capitalize() for part in parts[1:])
        translated[translated_key] = deepcopy(value)
    return translated


def _translate_distribution(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("AML distribution must be a mapping.")
    source = dict(value)
    raw_type = _property(source, "type", "distributionType")
    if not raw_type:
        raise ValueError("AML distribution has no type/distributionType.")
    distribution_type = _DISTRIBUTION_TYPES.get(str(raw_type).lower())
    if distribution_type is None:
        raise ValueError(f"Unsupported AML distribution type: {raw_type!r}")
    translated = _camelize_mapping(source)
    translated.pop("type", None)
    translated["distributionType"] = distribution_type
    return translated


def _iso8601_seconds(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.upper().startswith("P"):
            return normalized.upper()
        try:
            seconds = int(float(normalized))
        except ValueError as exc:
            raise ValueError(f"Unsupported AML timeout value: {value!r}") from exc
    else:
        seconds = int(value)
    if seconds <= 0:
        raise ValueError("AML timeout must be greater than zero.")
    return f"PT{seconds}S"


def _translate_limits(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("AML limits must be a mapping.")
    timeout = _property(value, "timeout", "timeout")
    if timeout is None:
        return None
    return {
        "jobLimitsType": "Command",
        "timeout": _iso8601_seconds(timeout),
    }


def _translate_services(
    services: Any,
    *,
    warnings: list[str],
) -> dict[str, dict[str, Any]] | None:
    if services is None:
        return None
    if not isinstance(services, Mapping):
        raise ValueError("AML services must be a mapping.")

    translated: dict[str, dict[str, Any]] = {}
    for raw_name, raw_service in services.items():
        name = str(raw_name)
        if not isinstance(raw_service, Mapping):
            warnings.append(f"Service {name!r}: skipped malformed service definition.")
            continue
        service = dict(raw_service)
        raw_type = _property(service, "type", "jobServiceType") or name
        normalized_type = str(raw_type).replace("_", "").lower()
        if normalized_type in _GENERATED_SERVICE_TYPES:
            warnings.append(
                f"Service {name!r}: skipped generated AML {raw_type!r} endpoint; "
                "Foundry will generate its own service links."
            )
            continue
        service_type = _SERVICE_TYPES.get(normalized_type)
        if service_type is None:
            raise ValueError(f"Unsupported AML job service type: {raw_type!r}")

        translated_service: dict[str, Any] = {"jobServiceType": service_type}
        if service.get("port") is not None:
            translated_service["port"] = int(service["port"])
        properties = service.get("properties")
        if isinstance(properties, Mapping) and properties:
            translated_service["properties"] = {
                str(key): str(item) for key, item in properties.items()
            }
        nodes = service.get("nodes")
        if isinstance(nodes, Mapping):
            node_type = _property(nodes, "type", "nodesValueType")
            if node_type and str(node_type).lower() == "all":
                translated_service["nodes"] = {"nodesValueType": "All"}
        translated[name] = translated_service
    return translated or None


def _translate_queue_settings(aml_job: Mapping[str, Any]) -> dict[str, Any] | None:
    queue = _copy_optional_mapping(aml_job, "queue_settings", "queueSettings") or {}
    job_tier = _property(queue, "job_tier", "jobTier") or _property(
        aml_job, "job_tier", "jobTier"
    )
    if not job_tier:
        return None
    normalized = str(job_tier).strip().capitalize()
    if normalized not in {"Null", "Spot", "Basic", "Standard", "Premium"}:
        raise ValueError(f"Unsupported AML job tier: {job_tier!r}")
    if normalized == "Null":
        return None
    return {"jobTier": normalized}


def translate_aml_command_job(
    aml_job: Mapping[str, Any],
    *,
    foundry_compute_id: str,
    foundry_instance_type: str,
    environment_image_reference: str,
    migrated_asset_ids: Mapping[str, str] | None = None,
    code_id: str | None = None,
    user_assigned_identity_id: str | None = None,
    model_input_path_suffixes: Mapping[str, str] | None = None,
    input_path_suffixes: Mapping[str, str] | None = None,
    output_asset_name_prefix: str | None = None,
    output_asset_version: str | None = None,
) -> TranslationResult:
    """Translate a materialized AML command job into a Foundry PUT body.

    Asset migration is intentionally explicit. ``migrated_asset_ids`` maps each
    original AML input URI to the new ``azureai://`` data/model ID, while
    ``code_id`` is the Foundry Dataset V3 asset created from the AML code snapshot.
    """

    if str(_property(aml_job, "type", "jobType") or "command").lower() != "command":
        raise ValueError("Only AML command jobs are supported.")
    command = str(aml_job.get("command") or "").strip()
    if not command:
        raise ValueError("AML command job has no command.")
    if not str(foundry_compute_id).strip():
        raise ValueError("foundry_compute_id must be non-empty.")
    if not str(foundry_instance_type).strip():
        raise ValueError("foundry_instance_type must be non-empty.")
    if not str(environment_image_reference).strip():
        raise ValueError("environment_image_reference must be non-empty.")

    warnings: list[str] = []
    warnings.extend(
        audit_aml_command_job_compatibility(
            aml_job,
            user_assigned_identity_id=user_assigned_identity_id,
        )
    )
    warnings.extend(
        _environment_compatibility_warnings(
            environment_image_reference,
            user_assigned_identity_id=user_assigned_identity_id,
        )
    )
    environment_variable_source = _property(
        aml_job,
        "environment_variables",
        "environmentVariables",
    )
    command, prepared_outputs, environment_variable_source = _prepare_default_output(
        command,
        aml_job.get("outputs"),
        environment_variable_source,
        warnings=warnings,
    )
    command, environment_variable_source = _prepare_file_output_paths(
        command,
        prepared_outputs,
        environment_variable_source,
        warnings=warnings,
    )
    command, environment_variable_source = _prepare_model_input_paths(
        command,
        environment_variable_source,
        model_input_path_suffixes or {},
        warnings=warnings,
    )
    command, environment_variable_source = _prepare_input_path_suffixes(
        command,
        environment_variable_source,
        input_path_suffixes or {},
        warnings=warnings,
    )
    command, environment_variables = _translate_templated_environment_variables(
        command,
        environment_variable_source,
        warnings=warnings,
    )
    source_name = str(aml_job.get("name") or "aml-command-job")
    asset_name_prefix = output_asset_name_prefix or f"migrated-{source_name}"
    asset_version = output_asset_version or create_foundry_asset_version()
    translated_inputs = _translate_inputs(
        aml_job.get("inputs"),
        migrated_asset_ids=migrated_asset_ids or {},
    )
    translated_outputs = _translate_outputs(
        prepared_outputs,
        asset_name_prefix=asset_name_prefix,
        asset_version=asset_version,
        warnings=warnings,
    )

    source_resources = _copy_optional_mapping(aml_job, "resources", "resources") or {}
    instance_count = _property(source_resources, "instance_count", "instanceCount") or 1
    properties: dict[str, Any] = {
        "jobType": "Command",
        "displayName": str(
            _property(aml_job, "display_name", "displayName") or source_name
        ),
        "description": str(aml_job.get("description") or ""),
        "experimentName": str(
            _property(aml_job, "experiment_name", "experimentName") or "Default"
        ),
        "command": command,
        "environmentImageReference": str(environment_image_reference),
        "computeId": str(foundry_compute_id),
        "resources": {
            "instanceCount": int(instance_count),
            "instanceType": str(foundry_instance_type),
        },
        "environmentVariables": environment_variables,
        "tags": {
            **{
                str(key): str(value)
                for key, value in dict(aml_job.get("tags") or {}).items()
            },
            "migration.source": "AzureML",
            "migration.sourceJob": source_name,
        },
        "inputs": translated_inputs,
        "outputs": translated_outputs,
    }
    portable_properties, _ = _portable_job_properties(aml_job.get("properties"))
    if portable_properties:
        properties["properties"] = portable_properties
    shared_memory_size = _property(source_resources, "shm_size", "shmSize")
    if shared_memory_size:
        properties["resources"]["shmSize"] = str(shared_memory_size)
    for aml_only_resource_key in ("docker_args", "dockerArgs", "locations"):
        if source_resources.get(aml_only_resource_key):
            warnings.append(
                f"Resource setting {aml_only_resource_key!r}: not copied because it "
                "is AML-compute-specific and unsupported by Foundry compute."
            )
    if code_id:
        properties["codeId"] = str(code_id)
    if user_assigned_identity_id:
        properties["userAssignedIdentityId"] = (
            str(user_assigned_identity_id).strip().rstrip("/")
        )

    distribution = _translate_distribution(aml_job.get("distribution"))
    if distribution:
        properties["distribution"] = distribution
    limits = _translate_limits(aml_job.get("limits"))
    if limits:
        properties["limits"] = limits
    queue_settings = _translate_queue_settings(aml_job)
    if queue_settings:
        properties["queueSettings"] = queue_settings
    priority = aml_job.get("priority")
    if priority:
        normalized_priority = str(priority).strip().capitalize()
        if normalized_priority not in {"Low", "Mid", "High"}:
            raise ValueError(f"Unsupported AML job priority: {priority!r}")
        properties["priority"] = normalized_priority
    services = _translate_services(aml_job.get("services"), warnings=warnings)
    if services:
        properties["services"] = services

    return TranslationResult(
        request_body={"properties": properties},
        warnings=tuple(warnings),
    )
