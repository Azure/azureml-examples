# AML command-job migration to Foundry compute

`aml-foundry-migrate` recreates an Azure Machine Learning command job on Foundry
compute. It migrates the job definition and, by default, the bytes behind its
code, data, and model dependencies. Dataset inputs can instead remain in source
storage through the optional zero-copy reference mode. Authentication comes from
the active Azure CLI login.

This is a Python CLI, not a PowerShell implementation. On Windows it is normally
invoked from PowerShell; `scripts/bootstrap_local_env.ps1` only prepares the
Python environment.

Pipeline, component, sweep, and AutoML jobs are intentionally outside this
workflow. The source and target are both single command jobs.

## End-to-end exercise

The `exercise` command proves the complete path rather than only translating a
JSON document. It performs these operations:

1. Creates local code, folder data, file data, an MLTable, and a custom model.
2. Registers the data and model assets in the source AML workspace.
3. Runs an AML command job on the selected AML compute.
4. Downloads the immutable AML code snapshot and registered model.
5. Runs one small AML export job for all data-like dependencies.
6. Uploads code and data through Foundry Dataset V3 using the project's standard
   storage connection.
7. Uploads and registers the model through Foundry Model V3.
8. Translates and submits the command job to Foundry compute.
9. Waits for the Foundry job to finish.
10. Downloads the Foundry folder, file, and MLTable outputs, compares their JSON
    content, and verifies the model output exists in Model V3.
11. Downloads AML and Foundry user-log evidence, extracts canonical application
    records, and requires the normalized logs and all four outputs to match.

```powershell
cd tools\foundrytrainingjob
.\scripts\bootstrap_local_env.ps1

az login
az account set --subscription <subscription-id>

aml-foundry-migrate exercise `
  --source-subscription <subscription-id> `
  --source-resource-group <aml-resource-group> `
  --source-workspace <aml-workspace> `
  --source-export-compute <aml-compute-name> `
  --project-endpoint https://<account>.services.ai.azure.com `
  --project-name <foundry-project> `
  --storage-connection <project-storage-connection> `
  --foundry-compute-id /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<account>/computes/<compute> `
  --foundry-instance-type Singularity.D4_v3 `
  --target-job-tier Premium `
  --user-assigned-identity-id /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.ManagedIdentity/userAssignedIdentities/<identity>
```

The project endpoint, project name, storage connection, Foundry compute ID,
instance type, and API version default to the values already configured by the
`foundrytrainingjob` tool. Source workspace arguments are required unless these
environment variables are set:

- `AML_MIGRATION_SOURCE_SUBSCRIPTION`
- `AML_MIGRATION_SOURCE_RESOURCE_GROUP`
- `AML_MIGRATION_SOURCE_WORKSPACE`
- `AML_MIGRATION_SOURCE_COMPUTE`
- `AML_MIGRATION_SOURCE_IDENTITY_DATASTORE` (defaults to
  `foundrymigrationidentityblob`)
- `AML_MIGRATION_DATASET_TRANSFER_MODE` (`upload` or `reference`)
- `AML_MIGRATION_SOURCE_STORAGE_CONNECTION` (required for `reference`)

## Formal live test

The same orchestration used by `aml-foundry-migrate exercise` is covered by a
dedicated live pytest. Configure the source workspace and run only this contract:

```powershell
$env:AML_MIGRATION_SOURCE_SUBSCRIPTION = "<subscription-id>"
$env:AML_MIGRATION_SOURCE_RESOURCE_GROUP = "<aml-resource-group>"
$env:AML_MIGRATION_SOURCE_WORKSPACE = "<aml-workspace>"
$env:AML_MIGRATION_SOURCE_COMPUTE = "<aml-compute-name>"
$env:AML_MIGRATION_E2E_WORK_DIR = "D:\migration-evidence\run-001" # optional

python -m pytest `
  tests\e2e\test_aml_command_job_migration.py `
  --run-live -vv -s -ra
```

Without `--run-live`, the test is skipped. With `--run-live` but no configured
source resource group, workspace, or compute, it also skips instead of creating
resources in an arbitrary workspace. Target settings use the existing
`FOUNDRY_TRAININGJOB__*` environment variables and default to Premium tier.

## Cost-bounded release qualification

The release gate combines every mutually compatible core behavior into one AML
source fixture, then migrates that same completed source twice:

- `upload`: URI-folder, URI-file, and MLTable bytes are copied into Foundry
  project Dataset V3 assets.
- `reference`: those same three AML inputs become Dataset V3 references through
  a Foundry connection whose target must match the AML source storage host.

Dataset V3 validates URI-folder references at the storage-container boundary.
For an AML data asset backed by a subfolder, migration registers the existing
container as the reference and appends the original relative blob prefix to
every command and templated-environment binding. The path visible to user code
therefore remains the AML asset folder; no bytes are copied. URI-file inputs
continue to reference their exact source blob.

The rich source job contains string, integer, number, and boolean literals;
URI-folder, URI-file, MLTable, and custom-model inputs; static and input-templated
environment variables; code, environment, compute, identity, shared memory, and
timeout behavior; and URI-folder, URI-file, MLTable, and custom-model outputs.
Both target runs must reproduce the AML application-log record and all four
logical output digests.

Foundry currently rejects the native `MLTable` output discriminator during
CommonRuntime output registration. Migration therefore preserves the output's
`MLTable` definition and payload files but registers the target as a
`uri_folder`. Analysis and release evidence report this as an adapted output,
not native MLTable asset-type parity.

```powershell
aml-foundry-migrate qualify-release `
  --source-subscription <subscription-id> `
  --source-resource-group <aml-resource-group> `
  --source-workspace <aml-workspace> `
  --source-export-compute <aml-compute-name> `
  --source-storage-connection <foundry-connection-to-aml-storage> `
  --project-endpoint https://<account>.services.ai.azure.com `
  --project-name <foundry-project> `
  --storage-connection <foundry-project-storage-connection> `
  --foundry-compute-id <foundry-compute-resource-id> `
  --foundry-instance-type <instance-type> `
  --user-assigned-identity-id <target-uai-resource-id> `
  --work-dir D:\migration-evidence\release-001
```

The worst case is four cloud jobs: one AML source job, one batched AML export,
and two Foundry target jobs. Pass `--existing-source-job` plus
`--existing-source-asset-version` to reuse a completed rich source fixture and
reduce that to three. Upload and reference target assets always receive distinct
versions, so the reference run cannot overwrite the copied datasets.

The formal live equivalent is:

```powershell
python -m pytest `
  tests\e2e\test_aml_command_job_release_e2e.py `
  --run-live -vv -s -ra
```

The evidence root contains per-mode `equivalence-report.json`, migration
manifests, analysis reports, downloaded raw logs/outputs, and one aggregate
`release-validation-report.json`. The aggregate report hashes every retained
artifact and each migration implementation file. It fails closed on terminal
status, parity, Dataset V3 type/wiring, connection, or runtime-RBAC mismatches.

A pass means the report's `qualifiedScope` is ready for a constrained preview.
It does not mean unrestricted customer release: `excludedScope` names pipeline,
sweep, AutoML, distributed, alternate delivery-mode, alternate model-format,
environment-build/private-registry, interactive-service, and advanced MLTable
or component-contract variants that still need their own live parity evidence.

## Component-by-component live validation

The isolated live matrix creates a different AML command job for each migration
component. Every case runs the source AML job, migrates it, runs the Foundry job,
downloads both registered outputs, and requires their JSON content to match. It
also asserts the component's translated Foundry request fields and writes a
`component-equivalence-report.json` under that case's temporary work directory.

```powershell
python -m pytest `
  tests\e2e\test_aml_command_job_component_migrations.py `
  --run-live -vv -s -ra
```

The matrix contains independent migrations for:

- code snapshot, environment image, target compute, identity, and URI-folder output
- string, integer, number, and boolean literal inputs
- URI-folder, URI-file, MLTable, and custom-model inputs
- static and input-templated environment variables
- URI-folder, URI-file, MLTable, and custom-model outputs
- shared-memory resources and command timeout
- descriptions, display/experiment names, tags, and portable properties
- source queue tier and priority
- SSH, JupyterLab, TensorBoard, and VS Code service definitions
- single-process MPI distribution
- zero-copy URI-folder, URI-file, and MLTable references

The three zero-copy cases skip unless
`AML_MIGRATION_SOURCE_STORAGE_CONNECTION` names a Foundry connection to the AML
source storage account. Run one component with `-k`, for example
`-k uri_file_input`. These tests are marked `live`, `aml_foundry_migration`, and
`aml_foundry_migration_component`; they never run in the default offline lane.
Set `AML_MIGRATION_E2E_WORK_DIR` to retain each report and its downloaded source
and target evidence under `components/<case>/<asset-version>/<run-id>/`.

The test fails unless both jobs reach `Completed`, both expose the canonical
application record, and the normalized `results`, `summary`, and `trained_model`
JSON outputs are identical. It writes `equivalence-report.json` under the work
directory for CI publication or local diagnosis.

## Log and output equivalence

Raw AML and Foundry logs are not expected to be byte-identical. Each platform
adds different timestamps, rank prefixes, bootstrap output, mount paths, and
runtime diagnostics. The fixture therefore emits one stable line:

```text
MIGRATION_FIXTURE_RECORD:{"enabled":true,"epochs":3,...}
```

The monitor downloads and retains the raw bundles, extracts every prefixed JSON
record, canonicalizes keys and whitespace, removes duplicate records, and then
compares SHA-256 digests. A missing, malformed, or different record fails the
test. `MIGRATION_FIXTURE_COMPLETED` must also be present on both sides.
The report also records each service's terminal status, raw log filenames, and
raw-file SHA-256 hashes. Those raw hashes are evidence pointers, not equality
assertions: platform wrappers are expected to differ.

Output paths and registered asset IDs necessarily differ, so comparison is by
logical output rather than physical path:

- `results/result.json`
- `summary/summary.json`
- `metrics_table/metrics.jsonl`
- `trained_model/model.json`

JSON is parsed and serialized canonically before digesting. The AML output is
the oracle, including AML's observed MLTable materialization. Foundry validation
first checks its content against that AML result and then the final equivalence
step compares every logical output again. Raw logs, raw outputs, normalized
values, and both digest sets are recorded in the report.

## Migrate an existing job

### Analyze before writing

Run the read-only analyzer before creating any Foundry assets or jobs:

```powershell
aml-foundry-migrate analyze `
  --source-subscription <subscription-id> `
  --source-resource-group <aml-resource-group> `
  --source-workspace <aml-workspace> `
  --source-job <job-name> `
  --analysis-policy migratable `
  --report-file .\migration-analysis.json
```

The analyzer reads AML job/asset metadata and Foundry connection metadata only.
It does not need source export compute and does not create assets, jobs,
connections, or role assignments. Every capability record includes:

- `support`: `supported`, `conditional`, `manual_action`, or `unsupported`
- `semanticFidelity`: `equivalent`, `adapted`, `lossy`, or `unknown`
- `selectedAction`: for example `translate`, `reference`, `copy`, `replace`,
  `transform`, `allocate`, `filter`, or `drop`
- `verification`: `live`, `unit`, `none`, or `not_applicable`
- `blocking`, `externalDependency`, `referenceable`, source/target values, and a
  concrete remediation when one is known

Choose an enforceable policy:

| Policy | Pass condition |
| --- | --- |
| `advisory` | Always exits `0`; inspect the report. |
| `migratable` | The concrete invocation has no unsupported or unresolved blockers. |
| `lossless` | Migratable, with no lossy or unknown source semantics. |
| `reference-only` | Migratable and every external dependency can remain in place. |
| `strict` | Lossless and every encountered capability has completed live verification. |

A policy failure still writes/prints the report but exits `2`. Operational or
authentication errors exit `1`. Use the same guard inline with migration:

```powershell
aml-foundry-migrate migrate `
  --source-resource-group <aml-resource-group> `
  --source-workspace <aml-workspace> `
  --source-export-compute <aml-compute-name> `
  --source-job <job-name> `
  --preflight-policy migratable
```

When preflight fails, migration never starts. For `reference-only`, the report
also validates that the supplied Foundry storage connection target host matches
each resolved AML data URI; an unrelated but valid storage connection does not
count.

### Permission and RBAC preflight

Analysis also performs a read-only runtime-identity/RBAC inspection. It resolves
the target UAI to its principal/client IDs, enumerates effective role assignments
including ancestor scopes, and evaluates only roles that provide the required
runtime data actions. Generic ARM `Contributor` is **not** treated as blob data
access.

The required checks are derived from the selected migration strategy:

| Scope | Required accepted roles |
| --- | --- |
| Foundry project | `Foundry User`, `Azure AI Developer`, or `Azure AI Administrator` |
| Foundry project storage | `Storage Blob Data Contributor` or `Storage Blob Data Owner` |
| Each AML source storage account in reference mode | `Storage Blob Data Reader`, `Storage Blob Data Contributor`, or `Storage Blob Data Owner` |
| Private environment ACR | `AcrPull`, `AcrPush`, or the ABAC repository Reader/Writer/Contributor roles |

Connection metadata supplies project and target-storage ARM scopes. Resolved AML
blob hosts supply source-storage scopes. Private ACR hosts are resolved to registry
resources in candidate source/target subscriptions. SAS/API-key connections do
not require UAI storage RBAC; AAD/managed-identity connections do.

Permission findings use four statuses:

- `satisfied`: an unconditional accepted assignment exists at the scope or an ancestor
- `conditional`: a matching assignment has an Azure RBAC condition; evaluate the condition manually
- `missing`: role assignments were readable, but no accepted assignment exists
- `unknown`: the identity/scope/role assignments could not be inspected, or a custom role's actions were not resolved

`missing` and `unknown` block `migratable` preflight. Inspection failure is never
misreported as a missing assignment. Reports include accepted roles, principal ID,
effective assignment scopes, conditions, and remediation. Summary fields include
`runtimePermissionsSatisfied`, `permissionCountsByStatus`,
`missingPermissionIds`, `unknownPermissionIds`, and
`conditionalPermissionIds`.

#### Opt-in source-storage access grant

The `migrate` command can repair analyzer-confirmed missing source-storage read
access before a reference migration:

```powershell
aml-foundry-migrate migrate `
  --source-resource-group <aml-resource-group> `
  --source-workspace <aml-workspace> `
  --source-export-compute <aml-compute-name> `
  --source-job <job-name> `
  --dataset-transfer-mode reference `
  --source-storage-connection <foundry-connection-to-aml-storage> `
  --user-assigned-identity-id <target-job-uai-resource-id> `
  --grant-reference-storage-access
```

This is an explicit RBAC mutation and is disabled by default. The flag:

- requires reference transfer and an explicit target job UAI
- requires that UAI to already be attached to the Foundry project
- runs `migratable` analysis before changing RBAC
- makes no change if any non-grantable migration blocker is present
- creates only `Storage Blob Data Reader` assignments at exact source storage-
  account scopes reported as `missing`
- does not override `unknown` or conditional findings
- uses deterministic role-assignment IDs so retries are idempotent
- reruns preflight after the assignment and before creating migration assets or
  submitting the Foundry job
- records created assignments under `referenceStorageRoleAssignments` in CLI
  output and atomically in `reference-storage-role-assignments.json` before
  post-grant preflight or migration starts

It never creates the Foundry source-storage connection and never grants Foundry
project, target-storage, private-ACR, owner, contributor, or role-administration
access. It also does not attach identities to the Foundry project. The signed-in
caller must already have
`Microsoft.Authorization/roleAssignments/write` on each exact source storage
scope. Azure RBAC propagation can still delay runtime access; the post-grant
preflight proves control-plane visibility, while the migrated job remains the
data-plane authority.

Static RBAC inspection cannot prove deny assignments, inactive PIM eligibility,
custom-role `actions`/`dataActions`, network ACLs, private DNS, service-side
connection policy, or eventual role propagation. It also does not prove the
**caller** can create AML export jobs or Foundry assets/jobs; source job/asset and
connection reads are exercised by analysis, while write authorization is only
fully proven by migration or a service-provided validate endpoint. Runtime E2E
tests remain the final authority for identity and networking.

The formal fixture requires project and target-storage checks in both modes and
source-storage reader access in reference mode. A missing source reader blocks
submission unless the explicit grant flag is used successfully.

Current external-asset disposition is explicit:

| Dependency | Current action | Reference status |
| --- | --- | --- |
| URI file/folder and MLTable bytes | `copy` or `reference` | Dataset V3 references are implemented and live-proven. |
| OCI environment image | `reference` | Reused in place through `environmentImageReference`; digest pinning is recommended. |
| Code snapshot | `copy` | Dataset V3 code references are a feasible follow-up, but not implemented/live-proven. |
| AML model | `copy` | Re-registered through Model V3; an external-storage reference contract is not yet proven. |
| Outputs | `allocate` | They are new results, so Foundry must write/register them rather than reference source outputs. |

This means `reference-only` intentionally fails a normal code-and-model job
today and names `asset.code` plus each model input in
`nonReferenceableDependencyIds`. It is a roadmap signal, not a false claim that
all bytes can already remain in place.

### Capability test contract

Every analyzer finding is assigned to one of the catalog families in
`CAPABILITY_FAMILY_CATALOG`. The analyzer fails closed if code attempts to emit a
finding whose family is not cataloged. Every report embeds the full catalog and
adds `family` plus `testCoverage` to each finding.

The exhaustive unit matrix is:

```powershell
python -m pytest `
  tests\unit\test_aml_command_job_capability_matrix.py `
  -q
```

That matrix enforces all of the following:

- every catalog family is emitted by a maximal command-job definition
- every catalog family names an existing owning unit test
- every declared live scenario or E2E evidence path exists
- analyzer and translator type/mode/distribution/service enums remain identical
- every literal, data, model, output, input mode, output mode, distribution, and
  service value is analyzed and translated
- MPI, PyTorch, TensorFlow, and Ray fields are mapped individually
- environment reference/tag/private-ACR/build-required variants are classified
- every unsupported type/mode/distribution/service branch blocks migration
- every analysis policy has a passing contract, while separate negative tests
  verify policy failure and no-write preflight behavior

RBAC has a dedicated matrix in
`tests/unit/test_aml_command_job_permissions.py`. It covers inherited scope,
read-versus-write data roles, generic Contributor rejection, ACR/Foundry role
sets, conditional assignments, custom-role uncertainty, pagination, identity
resolution, cross-subscription grouping, connection auth metadata, and the
distinction between missing and uninspectable assignments.

`unitTestCoverageComplete` in the report is therefore a guarded contract rather
than an informal claim. Live coverage remains deliberately separate:
`verification` describes the encountered variant, `liveEvidence` names existing
tests/scenarios for its family, and `familiesWithoutLiveEvidence` exposes gaps.
An unsupported or manual-action capability can be fully unit-tested while still
having no live migration path; the report does not relabel that as supported.

```powershell
aml-foundry-migrate migrate `
  --source-subscription <subscription-id> `
  --source-resource-group <aml-resource-group> `
  --source-workspace <aml-workspace> `
  --source-export-compute <aml-compute-name> `
  --source-job <job-name> `
  --project-endpoint https://<account>.services.ai.azure.com `
  --project-name <foundry-project> `
  --storage-connection <project-storage-connection> `
  --foundry-compute-id <foundry-compute-resource-id> `
  --foundry-instance-type <singularity-instance-type>
```

To avoid copying data inputs, register zero-copy Dataset V3 references instead:

```powershell
aml-foundry-migrate migrate `
  --source-subscription <subscription-id> `
  --source-resource-group <aml-resource-group> `
  --source-workspace <aml-workspace> `
  --source-export-compute <aml-compute-name> `
  --source-job <job-name> `
  --dataset-transfer-mode reference `
  --source-storage-connection <foundry-connection-to-aml-storage> `
  --project-endpoint https://<account>.services.ai.azure.com `
  --project-name <foundry-project> `
  --storage-connection <foundry-project-storage-connection> `
  --foundry-compute-id <foundry-compute-resource-id> `
  --foundry-instance-type <singularity-instance-type>
```

`reference` applies to `uri_file`, `uri_folder`, and MLTable-backed data inputs.
It resolves AML datastore paths to their HTTPS storage URIs and calls Dataset V3
`create_or_update` with `is_reference=true`; it does not run the AML data export
job and does not upload those bytes to Foundry project storage. The Foundry
connection must point to the source storage account, and the target job identity
must have data-plane read access. Every reference migration reads the connection
metadata and verifies that its target host matches every resolved data-input host
before creating Foundry assets. Code snapshots and models are still uploaded
because they use the distinct Dataset V3 code and Model V3 contracts. Output
assets are still written and registered by the Foundry job.

For URI-folder and MLTable inputs, the registered Dataset V3 URI is the AML
container root and `migration-manifest.json` records `registeredDataUri` plus
`foundryInputPathSuffix`; concatenating them must reproduce `referenceDataUri`.

Use `--environment-image` when the AML environment was built from a Docker
context or has a Conda overlay. Foundry accepts a runnable image reference; it
does not rebuild the AML environment definition. Use `--source-code-path` when
the source job does not expose a versioned workspace code asset, such as a
registry-backed or externally managed code snapshot.

## Primitive changes

| AML command-job primitive | Foundry representation | Migration behavior |
| --- | --- | --- |
| `code` workspace asset | Dataset V3 `codeId` | Downloads the immutable AML code version and uploads the folder. |
| AML environment asset | `environmentImageReference` | Resolves an image-only environment; rejects Conda/build definitions unless an image override is supplied. |
| AML compute name | Cognitive Services `computeId` plus `resources.instanceType` | Replaced by the target settings; AML VM size is not copied. |
| `string`, `integer`, `number`, `boolean` inputs | `jobInputType: literal` | Converts the value to the Foundry literal wire shape. |
| `uri_file`, `uri_folder`, `mltable` inputs | Dataset V3 `azureai://.../data/.../versions/...` | Defaults to exporting bytes through source compute and uploading them to project storage. Optional `reference` mode registers the existing source storage URI without copying bytes. |
| custom/MLflow/SafeTensors/Triton model inputs | Model V3 `azureai://.../models/.../versions/...` | Uses AML model download first; falls back to source-compute export, then uploads customer bytes under `model/`, registers the model, and targets that payload path from command placeholders. This keeps Model V3's `manifest.base.json` outside the path seen by the migrated command. |
| AML datastore output path | Foundry output `assetName` plus `assetVersion` | Drops the workspace datastore path and allocates a project-scoped output asset. AML MLTable outputs are registered as URI folders while retaining the `MLTable` file and payload. |
| input/output delivery modes | Foundry delivery modes | Maps input modes to their Pascal-cased equivalents. AML `upload` outputs use Foundry `ReadWriteMount`, the service's completed-job registration path for `uri_file`, `uri_folder`, and model outputs. |
| templated environment variable | shell assignment in `properties.command` | Foundry expands input/output placeholders in the command, not in `environmentVariables`. Static variables remain unchanged. |
| `limits.timeout` seconds | ISO-8601 `Command` limits | For example, `3600` becomes `PT3600S`. |
| AML distribution | Foundry distribution discriminator | Supports MPI, PyTorch, TensorFlow, and Ray fields. |
| SLA tier and queue priority | `resources.properties.AISuperComputer.slaTier` plus source queue fields | The CLI defaults the target SLA to Premium. `--target-job-tier Standard` is available when the target account has Standard quota; `Preserve` omits the target override. Source queue tier/priority remain separate translated fields when the source API exposes them. |
| user identity | `userAssignedIdentityId` | Uses the explicitly supplied target UAI. AML identity objects are not copied across resource boundaries. |
| generated Studio/Tracking service | Foundry-generated links | Omits AML-generated endpoints. Explicit SSH/Jupyter/TensorBoard/VS Code/Theia/Grafana/custom service definitions are translated. |

Descriptions, experiment name, tags, instance count, shared-memory size, named
inputs, named outputs, output descriptions/tags, distribution settings, timeout,
queue tier, priority, and supported interactive services are retained.

## Compatibility warnings

Before transferring assets, the migrator audits the materialized AML command
job and prints `WARNING:` records for source semantics that cannot be preserved.
The same warnings are retained in `migration-manifest.json` and returned by the
CLI. The audit currently covers:

- source identity replacement, parent-job lineage, notification settings, and
  non-default deterministic/caching semantics
- source instance type replacement and AML-only Docker/location settings
- fixed `path_on_compute`, input defaults/optionality/ranges/enums, explicit
  datastore selection, intellectual-property metadata, and early outputs
- alternate model formats that are translated but do not yet have live
  migration parity coverage; AML MLTable outputs use the live-validated
  URI-folder adaptation described above
- AML runtime-only properties, which are filtered while portable custom and
  MLflow lineage properties are retained
- mutable image tags and private-ACR identity/network prerequisites

Unsupported input/output types, modes, distribution types, priorities, service
types, or unresolved environments remain hard errors rather than warnings. This
prevents the tool from submitting a job whose behavior it already knows it
cannot represent.

## Environment migration

Foundry command jobs consume `environmentImageReference`, so image-backed AML
environments are reused **in place**. The migrator resolves a versioned AML
environment asset to its image reference and submits that OCI reference; it does
not download image layers or push them to another registry. This works for MCR,
public registries, and direct ACR references that Foundry compute can reach.

For private ACR images, the target UAI needs `AcrPull`, and Foundry compute must
have working DNS and network reachability to the registry and its data endpoint.
AML workspace identity, registry credentials, private endpoints, and firewall
rules are not copied. Tag-based references are accepted but produce a warning;
pin `repository@sha256:<digest>` when reproducibility matters.

Environment behavior beyond the image itself:

- static environment variables are copied to the Foundry job
- input/output-templated variables are moved into the shell command because
  Foundry expands placeholders there, not in `environmentVariables`
- AML Conda overlays and Docker build contexts are rejected; publish an
  equivalent image first and pass `--environment-image`
- AML environment tags, descriptive properties, build datastore, and asset
  version metadata are not runtime dependencies and are not recreated
- source compute image caches, Docker arguments, node locations, secrets,
  certificates, proxy configuration, and managed-network rules are not inferred
- GPU/CUDA/driver compatibility, CPU architecture, entrypoint, working directory,
  user, and native libraries remain properties of the referenced image and the
  chosen Foundry compute SKU

Copying an image to a target ACR is not currently part of the migration CLI.
That can be added as a separate `reference` versus `import` policy, but an import
must also handle cross-tenant authorization, multi-architecture manifests,
signatures/SBOMs, immutable digest mapping, private endpoints, and target-UAI
`AcrPull`. Reusing a digest-pinned source image is the lower-risk default.

## Coverage tiers

The formal live parity fixture proves all core features together, while the
component matrix reruns them as isolated migrations so one failed primitive has
a specific test identity. Together they cover a single-node command job with
code, all four primitive inputs, URI file/folder data, MLTable bytes, a custom
model, static and templated environment variables, timeout/shared memory, and
URI file/folder, MLTable, plus custom-model outputs. Upload and zero-copy Dataset V3 input
paths both have dedicated component cases. The matrix also defines isolated
cases for metadata, Standard/High scheduling, four interactive service types,
and single-process MPI. A case counts as live evidence only after it completes
against configured AML and Foundry resources; collection alone is not evidence.

The following are translated and unit-tested but still need migration-specific
live parity tests: multi-node MPI, PyTorch/TensorFlow/Ray distributions,
mount/direct/evaluation delivery modes, Jupyter/Theia/Grafana/custom services,
alternate queue/priority values, and alternate MLflow/SafeTensors/Triton model
formats. Ray's current SDK fields (`port`, `address`, dashboard settings, and
head/worker arguments) are preserved.

The following require an explicit policy rather than transparent translation:
nontrivial MLTable transformations, optional or constrained component inputs,
early-available outputs, source output destinations, job lineage/notifications,
secret-backed settings, and registry-backed environments or code assets.

## Asset-transfer behavior

Data is never copied by parsing datastore URLs or exporting credentials. The
script submits an AML staging command that binds each source URI using normal AML
input resolution, copies the mounted/downloaded content to named outputs, and
downloads those outputs through the AML SDK. This supports workspace data assets,
datastore URIs, registry or cross-workspace references accepted by the source
job, and external URIs that the source compute can read.

Source workspaces may disable storage-account shared keys. The exercise handles
that configuration without changing the account policy: fixture bytes and
staging outputs use `AzureCliCredential` against `workspaceblobstore`, and source
jobs run under AML user identity. Code uses AML's `startPendingUpload` handshake,
uploads to the service-assigned temporary container with Entra ID, stamps the
indicator blob with `upload_status=completed` plus asset name/version, and
registers the local content hash. This produces an AML-protected snapshot; merely
registering an arbitrary HTTPS blob folder is rejected by AML before user code
starts.

The formal live fixture uses download/upload delivery for source data and
outputs. On some classic AML compute stacks, `ro_mount`/`rw_mount` still enters
`UriMountSession` and fails with `ScriptExecution.StreamAccess.Authentication`
when shared keys are disabled, even when the submitting user has Blob Data
Contributor. Mount-mode translation remains covered by unit contract tests; use
download/upload for the live source oracle unless that AML workspace's mount
identity path has been configured and verified separately.

Foundry uploads use the target project's named storage connection. Temporary SAS
credentials returned by Dataset V3 or Model V3 remain in memory and are never
written to the manifest or console. Model V3 adds service metadata at the asset
root, so migrated model payloads are deliberately stored under `model/` and
model placeholders in the command are translated to that subdirectory.

## Resume and audit files

Each run writes under `--work-dir`:

- `migration-manifest.json`: source/target identity, sanitized source job,
  downloaded paths, uploaded asset IDs, export job, target job, warnings, status,
  and integrity fingerprints
- `foundry-job-request.json`: the translated Foundry request with secret-like
  values and signed query parameters redacted
- `source-code/`: downloaded AML code snapshot
- `inputs/`: downloaded registered models
- `export-code/` and `export-download/`: batched AML dependency export
- `validation/`: downloaded Foundry outputs for an `exercise` run
- `evidence/aml/`: AML stream log and keyless source-output downloads
- `evidence/foundry/`: Foundry `user_logs/*` and normalized output references
- `equivalence-report.json`: formal live-test result and comparison digests

When `--work-dir` is omitted, runs are stored under
`~/.aml-foundry-migration/runs/<job-name>`. Set `AML_MIGRATION_RUNS_DIR` to use
a different base directory. Migration manifests and saved request JSON are
sanitized before writing, but downloaded code, data, models, outputs, and raw
service logs can contain customer content. Treat the complete work directory as
sensitive evidence and do not place it in source control.

The manifest is updated atomically after each expensive operation. Rerunning the
same command with the same work directory reuses completed downloads, uploads,
the AML export job, and the submitted Foundry job. Resume is bound to fingerprints
of the complete correctness-relevant invocation, the materialized AML source-job
definition, each source URI, and the submitted Foundry request. Any mismatch is
rejected before stale assets or a stale job can be reused. Polling interval,
timeout, and wait/no-wait remain adjustable operational controls. Manifests from
older builds that lack complete resume fingerprints require a new work directory.

When waiting for completion, only Foundry `Completed` is successful. `Failed`,
`Canceled`, or `Cancelled` terminates the migration with a nonzero CLI exit code
while retaining the recorded status for diagnosis. `Paused` remains pollable
because live jobs can resume through `Paused` to `Running` and `Completed`.

Choose a new work directory to intentionally produce a second independent copy.
Output asset names include a run suffix because Foundry output versions are
immutable and reuse can return a conflict.

## Requirements and permissions

- Python 3.10 or newer with this package's dependencies installed
- Azure CLI authenticated to the source and target tenant
- AML workspace read/job-create permissions
- permission to use the selected AML compute and workspace storage
- permission to download AML code and model assets
- `Storage Blob Data Contributor` (or equivalent data-plane rights) on the AML
  workspace storage account when shared-key access is disabled
- Foundry project access, including Dataset V3 and Model V3 create/read
- access to the project storage connection
- permission to submit to the selected Foundry compute
- target UAI RBAC on private images or data when a UAI is supplied
- `Microsoft.Authorization/roleAssignments/write` on resolved AML source storage
  accounts only when `--grant-reference-storage-access` is requested

The workflow leaves source assets, source/export jobs, Foundry assets, and the
Foundry job in place for inspection. Delete them with the owning service's normal
retention or cleanup tooling after validation.

## Deliberate boundaries

- No pipelines, components, sweeps, AutoML, or pipeline-child reconstruction.
- AML Conda overlays and Docker build contexts require a prebuilt image override.
- AML compute-specific Docker arguments and location hints are reported but not
  copied because Foundry compute owns those settings.
- A non-versioned code reference requires `--source-code-path`.
- Secret values are not migrated. Recreate secret-backed settings as Foundry
  connections or target identity access rather than writing secrets into a job.
- Network-isolated source data must be readable by the selected AML export
  compute; network-isolated target assets must be readable by Foundry compute.
