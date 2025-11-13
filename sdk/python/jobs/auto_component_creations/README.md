# Azure ML Component Deployment System

This directory contains Azure ML components and pipelines, along with a deployment script to automatically register and deploy them.

## Directory Structure

```
Ignite_Component_Creation/
├── components_creation_index.json      # Index file defining all components and pipelines
├── deploy_components.py                # Main deployment script
├── chat_completion_model_import/       # Model import component
│   └── spec.yaml
├── eagle3_chat_completion_finetune/    # Eagle3 trainer component
│   └── spec.yaml
├── eagle3_chat_completion_pipeline/    # Eagle3 pipeline
│   └── spec.yaml
├── azure_reinforcement_learning_trainer/  # ARL trainer component
│   └── spec.yaml
├── azure_reinforcement_learning_pipeline/ # ARL pipeline
│   └── spec.yaml
└── sweep_checkpoint_evaluations/       # Checkpoint evaluation component and pipeline
    ├── sweep_checkpoint_evals_spec.yaml
    └── sweep_checkpoint_evals_pipeline.yaml
```

## Components Index (components_creation_index.json)

The index file contains:
- **registry_name**: Default registry name to use (test_centralus)
- **components**: Array of component/pipeline definitions with:
  - `name`: Component name
  - `type`: "component", "trainer", or "pipeline"
  - `path`: Relative path to spec.yaml file
  - `description`: Human-readable description
  - `pipeline`: (For trainers) Name of the pipeline that uses this trainer
  - `trainers`: (For pipelines) Array of trainer component names
  - `dependencies`: (For pipelines) Array of dependency component names

### Trainer-Pipeline Pairs

The system recognizes trainer-pipeline pairs through:
1. Trainers have a `pipeline` field pointing to their pipeline
2. Pipelines have a `trainers` array listing their trainer components

Current pairs:
- **Eagle3**: `eagle3_training` trainer ↔ `eagle3_chat_completion_pipeline`
- **ARL**: `arl_trainer_component` trainer ↔ `arl_finetune_pipeline`

## Deployment Script (deploy_components.py)

The script automates the entire deployment process:

### Features

1. **Version Management**: Automatically bumps versions for all components
2. **Registry Updates**: Updates all registry references to the target registry
3. **Dependency Resolution**: Deploys components in topological order based on dependencies
4. **Reference Updates**: Updates pipeline references to newly deployed component versions
5. **Flexible Targeting**: Deploy to registry or workspace

### Usage Examples

#### Deploy everything to default registry
```bash
python deploy_components.py
```

#### Deploy to a specific registry
```bash
python deploy_components.py --registry-name my-registry
```

#### Deploy to a workspace
```bash
python deploy_components.py --resource-group my-rg --workspace-name my-workspace
```

#### Deploy specific components only
```bash
python deploy_components.py --components chat_completion_model_import eagle3_training
```

#### Deploy specific pipelines only
```bash
python deploy_components.py --pipelines eagle3_chat_completion_pipeline
```

#### Dry run (preview changes without deploying)
```bash
python deploy_components.py --dry-run
```

### Command-Line Arguments

- `--registry-name, --reg`: Azure ML registry name (overrides value from index.json)
- `--resource-group, --rg`: Azure resource group name (use with --workspace-name)
- `--workspace-name, --ws`: Azure ML workspace name (use with --resource-group)
- `--components`: List of specific component names to deploy
- `--pipelines`: List of specific pipeline names to deploy
- `--dry-run, --dry`: Print changes without executing deployment
- `--index-file`: Path to index JSON file (default: components_creation_index.json)

## How It Works

### Phase 1: Component Deployment

1. Reads the components_creation_index.json file
2. Filters components (type != "pipeline")
3. Performs topological sort based on dependencies
4. For each component:
   - Reads current version from spec.yaml
   - Bumps version (0.0.0 → 0.0.1, 0.0.1.y4 → 0.0.1.y5)
   - Updates registry references to target registry
   - Writes updated spec.yaml
   - Deploys component using `az ml component create`
   - Tracks version for pipeline updates

### Phase 2: Pipeline Deployment

1. Filters pipelines (type == "pipeline")
2. Performs topological sort based on dependencies
3. For each pipeline:
   - Reads current version from spec.yaml
   - Bumps version
   - Updates trainer component references with new versions
   - Updates dependency component references with new versions
   - Updates registry references to target registry
   - Writes updated spec.yaml
   - Deploys pipeline using `az ml component create`

## Version Bumping

The script supports two version formats:

1. **Standard**: `0.0.0` → `0.0.1` → `0.0.2`
2. **Custom**: `0.0.1.y4` → `0.0.1.y5` → `0.0.1.y6`

## Registry Reference Format

All registry references follow the format:
```
azureml://registries/{registry_name}/components/{component_name}/versions/{version}
```

The script ensures all references use the target registry name consistently.

## Prerequisites

- Azure CLI installed and authenticated (`az login`)
- Azure ML CLI extension installed (`az extension add -n ml`)
- Appropriate permissions to create components in the target registry/workspace

## Comparison with deploy.py

This script (`deploy_components.py`) extends the functionality of the reference `C:\batch_utilities\deploy.py`:

### Key Differences

| Feature | deploy.py | deploy_components.py |
|---------|-----------|---------------------|
| Configuration | Command-line args | JSON index file |
| Scope | Single trainer-pipeline pair | Multiple components and pipelines |
| Dependencies | Manual ordering | Automatic topological sort |
| Component Discovery | Specified paths | Auto-discovered from index |
| Trainer-Pipeline Pairing | Implicit | Explicit in index |
| Registry Updates | Per-component | Global with tracking |

### Advantages

1. **Scalability**: Easily manage dozens of components without complex CLI args
2. **Documentation**: Index file serves as component catalog
3. **Automation**: Dependency resolution eliminates manual ordering
4. **Maintainability**: Add new components by updating JSON, not script logic
5. **Traceability**: Clear relationships between trainers, pipelines, and dependencies

## Troubleshooting

### Error: "Circular dependency detected"
- Check the `dependencies` fields in components_creation_index.json
- Ensure no circular references exist

### Error: "Could not find version in spec"
- Verify all spec.yaml files have a `version:` field
- Check YAML formatting

### Error: "Could not find component reference"
- Ensure pipeline spec.yaml files reference components with full azureml:// paths
- Check component names match exactly

### Authentication Issues
- Run `az login` to authenticate
- Verify you have permissions: `az ml registry show --name test_centralus`

## Extending the System

### Adding a New Component

1. Create component directory with spec.yaml
2. Add entry to components_creation_index.json:
```json
{
  "name": "my_new_component",
  "type": "component",
  "path": "my_new_component\\spec.yaml",
  "description": "My new component"
}
```
3. Run deployment script

### Adding a New Trainer-Pipeline Pair

1. Create trainer and pipeline directories with spec.yaml files
2. Add trainer entry:
```json
{
  "name": "my_trainer",
  "type": "trainer",
  "path": "my_trainer\\spec.yaml",
  "description": "My trainer",
  "pipeline": "my_pipeline"
}
```
3. Add pipeline entry:
```json
{
  "name": "my_pipeline",
  "type": "pipeline",
  "path": "my_pipeline\\spec.yaml",
  "description": "My pipeline",
  "trainers": ["my_trainer"],
  "dependencies": ["chat_completion_model_import"]
}
```
4. Run deployment script

## Reverting Version Changes

If you need to undo version bumps, use the `revert_versions.py` script:

### Basic Usage

```bash
# Decrement all versions by 1 (preview)
python revert_versions.py --dry-run

# Apply the revert
python revert_versions.py

# Set all versions to 0.0.0
python revert_versions.py --version 0.0.0

# Revert specific components only
python revert_versions.py --components eagle3_training
```

### How It Works

The revert script:
1. Reads the components index
2. Finds all spec.yaml files
3. Decrements versions (0.0.5 → 0.0.4) or sets a specific version
4. Updates spec.yaml files (with --dry-run support)

**Note**: This only reverts version numbers in spec.yaml files. It does not:
- Delete deployed components from the registry/workspace
- Revert registry references in pipeline specs
- Undo other changes made during deployment

## Best Practices

1. **Always test with --dry-run first** to preview changes
2. **Commit spec.yaml changes** after successful deployment for version tracking
3. **Use descriptive names** in the index file for better documentation
4. **Keep dependencies minimal** to simplify deployment order
5. **Verify registry access** before bulk deployments
6. **Use revert_versions.py** if you need to undo version bumps before re-deploying

## Available Scripts

| Script | Purpose | Key Features |
|--------|---------|--------------|
| `deploy_components.py` | Deploy components and pipelines | Version bumping, dependency resolution, registry updates |
| `revert_versions.py` | Revert version changes | Decrement versions, set specific version |

## Support

For issues or questions:
- Check the script output for detailed error messages
- Review the Azure ML documentation: https://aka.ms/azureml/components
- Examine the reference deploy.py script: C:\batch_utilities\deploy.py
