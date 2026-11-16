---
page_type: sample
languages:
- python
products:
- azure-machine-learning
description: Analyze and migrate Azure Machine Learning command jobs to Microsoft Foundry Jobs.
---

# Migrate an Azure Machine Learning command job to Microsoft Foundry

## Overview

This overview introduces an analyzer-first migration workflow.

This preview CLI analyzes and migrates Azure Machine Learning command jobs to
Microsoft Foundry Jobs. It copies code and model assets, either copies or
references supported data assets, creates new Foundry outputs, and records a
sanitized, resume-safe migration manifest.

Use the
[guided notebook](migrate-command-job-to-foundry.ipynb) or invoke the CLI
directly.

## Objective

The objective is to migrate a supported command job without silently changing
its runtime behavior.

This sample shows how to:

- run read-only compatibility and RBAC analysis before creating resources;
- migrate supported command-job code, data, models, environment bindings,
  compute, identity, and outputs;
- choose copied data assets or zero-copy references to source storage; and
- resume interrupted work from sanitized migration manifests.

## Programming languages

The programming languages used by this sample are:

- Python

## Estimated runtime

The estimated runtime for setup and analysis is about 5 minutes.

Allow about 5 minutes to install and analyze an existing job. Migration time
depends on asset sizes, compute availability, and the target job runtime.

> [!IMPORTANT]
> This is a preview sample for standalone, single-instance command jobs. Run
> `analyze` first and review all adaptations, unsupported capabilities, and
> permission findings before migration.

## Install

Use Python 3.10 or later. The pinned Azure AI Projects preview build is hosted
on the Azure SDK public development feed.

```bash
python -m pip install \
  --extra-index-url https://pkgs.dev.azure.com/azure-sdk/public/_packaging/azure-sdk-for-python/pypi/simple \
  -e .
```

Authenticate with Azure CLI before running the tool:

```bash
az login
az account set --subscription <subscription-id>
```

## Use

Start with the read-only analyzer:

```bash
aml-foundry-migrate analyze \
  --source-subscription <subscription-id> \
  --source-resource-group <aml-resource-group> \
  --source-workspace <aml-workspace> \
  --source-job <job-name> \
  --project-endpoint https://<account>.services.ai.azure.com \
  --project-name <foundry-project> \
  --storage-connection <project-storage-connection> \
  --foundry-compute-id <foundry-compute-resource-id> \
  --user-assigned-identity-id <target-uai-resource-id>
```

Run `migrate` with the same source and target arguments after reviewing the
analysis. Data inputs use copy mode by default. Add
`--dataset-transfer-mode reference` and `--source-storage-connection` for
zero-copy references to source storage.

RBAC inspection is read-only by default. The optional
`--grant-reference-storage-access` switch grants Storage Blob Data Reader only
at the exact source-storage scope required by a reference migration and writes
sanitized audit evidence.

See [the migration guide](docs/AML_COMMAND_JOB_MIGRATION.md) for the complete
support matrix, adaptations, security model, resume behavior, and release
qualification workflow.

## Release status

The qualified scope is standalone, single-instance command jobs. Distributed
jobs, pipelines, sweeps, AutoML, environment builds, interactive services, and
other capabilities marked unsupported by the analyzer require manual work.

This sample contains no deployment configuration, live-run evidence, notebook
output, or credentials. Supply all Azure resource identifiers at runtime.