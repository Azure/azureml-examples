# Quick Reference Guide

## Common Commands

### Deploy Everything
```bash
# To default registry (test_centralus)
python deploy_components.py

# Preview changes first
python deploy_components.py --dry-run

# To a different registry
python deploy_components.py --registry-name my-registry

# To a workspace
python deploy_components.py --resource-group my-rg --workspace-name my-ws
```

### Deploy Specific Components

```bash
# Deploy only model import component
python deploy_components.py --components chat_completion_model_import

# Deploy multiple specific components
python deploy_components.py --components eagle3_training arl_trainer_component

# Deploy only pipelines
python deploy_components.py --pipelines eagle3_chat_completion_pipeline arl_finetune_pipeline
```

## Reverting Version Bumps

If you need to undo version changes:

```bash
# Decrement all versions by 1 (0.0.5 -> 0.0.4)
python revert_versions.py --dry-run
python revert_versions.py

# Set all components to specific version
python revert_versions.py --version 0.0.0

# Revert specific components
python revert_versions.py --components eagle3_training arl_trainer_component

# Revert specific pipelines
python revert_versions.py --pipelines eagle3_chat_completion_pipeline
```

### Revert Use Cases

**After accidental deployment:**
```bash
python revert_versions.py --version 0.0.0
```

**Testing version bumps:**
```bash
# Bump
python deploy_components.py --dry-run

# Revert
python revert_versions.py --dry-run
```

**Selective rollback:**
```bash
# Only revert trainers
python revert_versions.py --components eagle3_training arl_trainer_component
```

## Complete Workflow Examples

### 1. Fresh Deployment
```bash
# Preview everything
python deploy_components.py --dry-run

# Deploy when ready
python deploy_components.py
```

### 1. Fresh Deployment
```bash
# Preview everything
python deploy_components.py --dry-run

# Deploy when ready
python deploy_components.py

# Verify
az ml component list --registry-name test_centralus
```

#### 2. Update and Redeploy
```bash
# Update and deploy just one trainer
python deploy_components.py --components eagle3_training

# Then update its pipeline
python deploy_components.py --pipelines eagle3_chat_completion_pipeline
```

#### 2. Update and Redeploy
```bash
# Modify spec.yaml files
# Deploy changes (versions auto-bump)
python deploy_components.py

# Or deploy just what changed
python deploy_components.py --components eagle3_training
python deploy_components.py --pipelines eagle3_chat_completion_pipeline
```

#### 3. Rollback After Issues
```bash
# Eagle3 pair
python deploy_components.py --components chat_completion_model_import eagle3_training
python deploy_components.py --pipelines eagle3_chat_completion_pipeline

# ARL pair
python deploy_components.py --components chat_completion_model_import arl_trainer_component
python deploy_components.py --pipelines arl_finetune_pipeline
```

## Component Index Structure

### Component Types

1. **component**: Standalone component (e.g., model import, evaluations)
2. **trainer**: Training component (pairs with a pipeline)
3. **pipeline**: Pipeline component (orchestrates other components)

### Defining Relationships

#### Trainer → Pipeline
```json
{
  "name": "my_trainer",
  "type": "trainer",
  "path": "my_trainer\\spec.yaml",
  "pipeline": "my_pipeline"
}
```

#### Pipeline → Trainers
```json
{
  "name": "my_pipeline",
  "type": "pipeline",
  "path": "my_pipeline\\spec.yaml",
  "trainers": ["my_trainer"],
  "dependencies": ["chat_completion_model_import"]
}
```

## Version Formats

- **Standard**: `0.0.0` → `0.0.1` → `0.0.2`
- **Custom**: `0.0.1.y4` → `0.0.1.y5`

## Registry Reference Format

```yaml
component: azureml://registries/{registry_name}/components/{component_name}/versions/{version}
```

Example:
```yaml
component: azureml://registries/test_centralus/components/chat_completion_model_import/versions/0.0.1
```

## Deployment Order

The script automatically resolves dependencies. For the current setup:

1. **Independent Components**:
   - `chat_completion_model_import`
   - `eagle3_training`
   - `arl_trainer_component`
   - `sweep_checkpoint_evals`

2. **Pipelines** (deployed after components):
   - `arl_finetune_pipeline` (depends on: chat_completion_model_import, arl_trainer_component)
   - `eagle3_chat_completion_pipeline` (depends on: chat_completion_model_import, eagle3_training)
   - `sweep_checkpoint_evals_pipeline` (depends on: sweep_checkpoint_evals)

## Trainer-Pipeline Pairs

| Trainer | Pipeline | Shared Dependencies |
|---------|----------|---------------------|
| `eagle3_training` | `eagle3_chat_completion_pipeline` | `chat_completion_model_import` |
| `arl_trainer_component` | `arl_finetune_pipeline` | `chat_completion_model_import` |

## Troubleshooting Quick Fixes

### "Circular dependency detected"
Check `dependencies` in components_creation_index.json - remove circular references.

### "Could not find version"
Ensure all spec.yaml files have `version:` field at the top level.

### "Authentication failed"
```bash
az login
az account set --subscription <subscription-id>
```

### "Registry not found"
```bash
az ml registry show --name test_centralus
```

### Component reference not updated
Check that pipeline spec uses full path format:
```yaml
component: azureml://registries/test_centralus/components/component_name/versions/X.X.X
```

## File Checklist

- [ ] `components_creation_index.json` - Component definitions
- [ ] `deploy_components.py` - Deployment script
- [ ] Component `spec.yaml` files - Component specifications
- [ ] Pipeline `spec.yaml` files - Pipeline specifications
- [ ] `README.md` - Full documentation

## Example: Adding New Component

1. **Create directory and spec.yaml**:
```
my_new_component/
└── spec.yaml
```

2. **Add to index**:
```json
{
  "name": "my_new_component",
  "type": "component",
  "path": "my_new_component\\spec.yaml",
  "description": "My new component"
}
```

3. **Deploy**:
```bash
python deploy_components.py --components my_new_component --dry-run
python deploy_components.py --components my_new_component
```

## Reverting Version Bumps

If you need to undo version changes:

```bash
# Decrement all versions by 1 (0.0.5 -> 0.0.4)
python revert_versions.py --dry-run
python revert_versions.py

# Set all components to specific version
python revert_versions.py --version 0.0.0

# Revert specific components
python revert_versions.py --components eagle3_training arl_trainer_component

# Revert specific pipelines
python revert_versions.py --pipelines eagle3_chat_completion_pipeline
```

## Script Output

### Successful Deployment
```
================================================================================
Azure ML Component Deployment Script
================================================================================
Target: Registry (test_centralus)

PHASE 1: Deploying Components
  ✓ Deployed chat_completion_model_import version 0.0.1

PHASE 2: Deploying Pipelines
  ✓ Deployed eagle3_chat_completion_pipeline version 0.0.1

✓ Deployment completed successfully!
================================================================================
```

### Dry Run
```
[DRY RUN] Would update spec.yaml
[DRY RUN] Running: az ml component create ...
[DRY RUN] No changes were made. Run without --dry-run to deploy.
```
