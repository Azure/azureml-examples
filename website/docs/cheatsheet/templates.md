# Templates

## Introduction

Cookiecutter is a simple command-line tool that allows you to quickly create
new projects from pre-defined templates. Let's see it in action!

First go ahead and get cookiecutter using your environment manager of choice,
for example:

```bash
pip install cookiecutter
```

Then give this repo a home

```bash
cd ~/repos # or wherever your repos call home :-)
git clone <this-repo>
```

To create a new project from the `ScriptRunConfig` template for example, simply
run

```bash
cookiecutter path/to/cheatsheet/repo/templates/ScriptRunConfig
```

See [ScriptRunConfig](#ScriptRunConfig) for more details for this template.

## Project Templates

- ScriptRunConfig: Create a project to run a script in AML making use of the
ScriptRunConfig class. This template is well suited for smaller projects and
is especially handy for testing.

### ScriptRunConfig

[Cookiecutter](https://cookiecutter.readthedocs.io/en/1.7.2/README.html)
template for setting up an AML
[ScriptRunConfig](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
used to run your script in Azure.

#### Usage

Run the cookiecutter command

```bash
cookiecutter <path/to/cookiecutter/templates>/ScriptRunConfig
```

to create a new `ScriptRunConfig` project.

**Note.** Install with `pip install cookiecutter` (see
[cookiecutter docs](https://cookiecutter.readthedocs.io/en/1.7.2/installation.html)
for more installation options)

You will be prompted for the following:

- `directory_name`: The desired name of the directory (default:
"aml-src-script")
- `script_name`: The name of the python script to be run in Azure (default:
"script")
- `subscription_id`: Your Azure Subscription ID
- `resource_group`: Your Azure resource group name
- `workspace_name`: Your Azure ML workspace name
- `compute_target_name`: The name of the Azure ML compute target to run the
script on (default: "local", will run on your box)

Cookiecutter creates a new project with the following layout.

```bash
{directory_name}/
    {script_name}.py      # the script you want to run in the cloud
    run.py                # wraps your script in ScriptRunConfig to send to Azure
    config.json           # your Azure ML metadata
    readme.md             # this readme file!
```

See
[ScriptRunConfig](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
for more options and details on configuring runs.
