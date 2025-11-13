#!/usr/bin/env python3
"""
Script to deploy Azure ML components and pipelines from a JSON index.

This script:
1. Reads the components_creation_index.json file
2. Registers the registry name in all component/pipeline specs
3. Deploys components in dependency order
4. Updates pipeline references to newly deployed component versions
5. Deploys pipelines

Usage:
    # Deploy to registry (uses registry from index.json)
    python deploy_components.py [--dry-run]

    # Deploy to registry (override registry)
    python deploy_components.py --registry-name REGISTRY [--dry-run]

    # Deploy to workspace
    python deploy_components.py --resource-group RG --workspace-name WORKSPACE [--dry-run]

    # Deploy specific components only
    python deploy_components.py --components eagle3_training arl_trainer_component [--dry-run]

    # Deploy specific pipelines only
    python deploy_components.py --pipelines eagle3_chat_completion_pipeline [--dry-run]

Arguments:
    --registry-name: Azure ML registry name (overrides value from index.json)
    --resource-group: Azure resource group name (use with --workspace-name)
    --workspace-name: Azure ML workspace name (use with --resource-group)
    --components: List of specific component names to deploy
    --pipelines: List of specific pipeline names to deploy
    --dry-run: Print changes without executing deployment
    --index-file: Path to index JSON file (default: components_creation_index.json)
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def read_file(file_path):
    """Read file content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(file_path, content):
    """Write content to file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def bump_version(version_str):
    """
    Bump the version string.
    Handles versions like '0.0.1.y4' -> '0.0.1.y5' and '0.0.0' -> '0.0.1'

    Args:
        version_str: Current version string

    Returns:
        New version string
    """
    # Match pattern like 0.0.1.y4
    match = re.match(r'^(\d+\.\d+\.\d+\.y)(\d+)$', version_str)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        return f"{prefix}{number + 1}"

    # Match pattern like 0.0.1
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)$', version_str)
    if match:
        major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return f"{major}.{minor}.{patch + 1}"

    raise ValueError(f"Unsupported version format: {version_str}")


def get_version_from_spec(spec_path):
    """Extract version from spec.yaml file."""
    content = read_file(spec_path)
    version_match = re.search(r'^version:\s*(.+)$', content, re.MULTILINE)
    if not version_match:
        raise ValueError(f"Could not find version in {spec_path}")
    return version_match.group(1).strip()


def get_component_name_from_spec(spec_path):
    """Extract component name from spec.yaml file."""
    content = read_file(spec_path)
    name_match = re.search(r'^name:\s*(.+)$', content, re.MULTILINE)
    if not name_match:
        raise ValueError(f"Could not find name in {spec_path}")
    return name_match.group(1).strip()


def update_version_in_spec(spec_path, new_version):
    """Update version in a spec.yaml file."""
    content = read_file(spec_path)
    updated_content = re.sub(
        r'^version:\s*.+$',
        f'version: {new_version}',
        content,
        count=1,
        flags=re.MULTILINE
    )
    return updated_content


def update_registry_references(spec_path, registry_name, component_versions):
    """
    Update registry references in spec.yaml to use specified registry and latest versions.
    
    Args:
        spec_path: Path to spec.yaml
        registry_name: Registry name to use
        component_versions: Dict mapping component names to their versions
    
    Returns:
        Updated content
    """
    content = read_file(spec_path)
    
    # Pattern: azureml://registries/OLD_REGISTRY/components/COMPONENT_NAME/versions/VERSION
    pattern = r'(azureml://registries/)([^/]+)(/components/)([^/]+)(/versions/)([0-9.a-zA-Z]+)'
    
    def replace_registry(match):
        component_name = match.group(4)
        # Update registry and use new version if available
        if component_name in component_versions:
            new_version = component_versions[component_name]
            return f'{match.group(1)}{registry_name}{match.group(3)}{component_name}{match.group(5)}{new_version}'
        else:
            # Just update registry, keep version
            return f'{match.group(1)}{registry_name}{match.group(3)}{component_name}{match.group(5)}{match.group(6)}'
    
    updated_content = re.sub(pattern, replace_registry, content)
    return updated_content


def update_component_reference_in_pipeline(pipeline_spec_path, component_name, new_version, registry_name):
    """
    Update component version reference in pipeline spec.
    
    Args:
        pipeline_spec_path: Path to pipeline spec.yaml
        component_name: Name of the component to update
        new_version: New version string
        registry_name: Registry name to use
    
    Returns:
        Updated content
    """
    content = read_file(pipeline_spec_path)
    
    # Pattern: component: azureml://registries/REGISTRY/components/COMPONENT_NAME/versions/VERSION
    pattern = f'(component:\\s+azureml://registries/)([^/]+)(/components/{component_name}/versions/)([0-9.a-zA-Z]+)'
    replacement = f'\\g<1>{registry_name}\\g<3>{new_version}'
    updated_content = re.sub(pattern, replacement, content)
    
    return updated_content


def run_command(command, dry_run=False):
    """Run a shell command."""
    if isinstance(command, list):
        command = [str(arg) for arg in command]
        command_str = ' '.join(command)
    else:
        command_str = command

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running: {command_str}")

    if dry_run:
        return None

    is_windows = os.name == 'nt'

    if is_windows:
        result = subprocess.run(command_str, shell=True, capture_output=True, text=True)
    else:
        if isinstance(command, str):
            command = command.split()
        result = subprocess.run(command, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        print(f"ERROR: Command failed with exit code {result.returncode}", file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        sys.exit(1)

    return result


def build_dependency_graph(components_config):
    """
    Build a dependency graph from components configuration.
    
    Returns:
        Tuple of (dependency_graph, reverse_graph)
    """
    dependency_graph = {}
    reverse_graph = {}
    
    for comp in components_config:
        name = comp['name']
        dependency_graph[name] = comp.get('dependencies', [])
        reverse_graph[name] = []
    
    # Build reverse graph
    for name, deps in dependency_graph.items():
        for dep in deps:
            if dep in reverse_graph:
                reverse_graph[dep].append(name)
    
    return dependency_graph, reverse_graph


def topological_sort(components_config):
    """
    Perform topological sort on components based on dependencies.
    
    Returns:
        List of component names in deployment order
    """
    dependency_graph, _ = build_dependency_graph(components_config)
    
    # Count in-degrees
    in_degree = {comp['name']: 0 for comp in components_config}
    for name, deps in dependency_graph.items():
        for dep in deps:
            if dep in in_degree:
                in_degree[name] += 1
    
    # Queue of nodes with no dependencies
    queue = [name for name, degree in in_degree.items() if degree == 0]
    result = []
    
    while queue:
        # Sort to ensure deterministic order
        queue.sort()
        node = queue.pop(0)
        result.append(node)
        
        # Find all components that depend on this one
        for comp in components_config:
            if node in comp.get('dependencies', []):
                in_degree[comp['name']] -= 1
                if in_degree[comp['name']] == 0:
                    queue.append(comp['name'])
    
    if len(result) != len(components_config):
        raise ValueError("Circular dependency detected in components")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Deploy Azure ML components and pipelines from index'
    )
    parser.add_argument(
        '--registry-name', '--reg',
        type=str,
        help='Azure ML registry name (overrides value from index.json)'
    )
    parser.add_argument(
        '--resource-group', '--rg',
        type=str,
        help='Azure resource group name (use with --workspace-name for workspace deployment)'
    )
    parser.add_argument(
        '--workspace-name', '--ws',
        type=str,
        help='Azure ML workspace name (use with --resource-group for workspace deployment)'
    )
    parser.add_argument(
        '--components',
        type=str,
        nargs='+',
        help='List of specific component names to deploy'
    )
    parser.add_argument(
        '--pipelines',
        type=str,
        nargs='+',
        help='List of specific pipeline names to deploy'
    )
    parser.add_argument(
        '--dry-run', '--dry',
        action='store_true',
        help='Print changes without executing deployment'
    )
    parser.add_argument(
        '--index-file',
        type=str,
        default='components_creation_index.json',
        help='Path to index JSON file (default: components_creation_index.json)'
    )

    args = parser.parse_args()

    # Read index file
    script_dir = Path(__file__).parent
    index_path = script_dir / args.index_file
    
    if not index_path.exists():
        print(f"ERROR: Index file not found at {index_path}", file=sys.stderr)
        sys.exit(1)

    with open(index_path, 'r', encoding='utf-8') as f:
        index_data = json.load(f)

    # Determine deployment target
    if args.resource_group and args.workspace_name:
        deployment_target = 'workspace'
        registry_name = None
    elif args.registry_name:
        deployment_target = 'registry'
        registry_name = args.registry_name
    elif 'registry_name' in index_data:
        deployment_target = 'registry'
        registry_name = index_data['registry_name']
    else:
        print("ERROR: No registry name found. Provide --registry-name or set registry_name in index.json", 
              file=sys.stderr)
        sys.exit(1)

    components_config = index_data.get('components', [])
    
    if not components_config:
        print("ERROR: No components found in index file", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("Azure ML Component Deployment Script")
    print("=" * 80)
    if deployment_target == 'workspace':
        print(f"Target: Workspace ({args.resource_group}/{args.workspace_name})")
    else:
        print(f"Target: Registry ({registry_name})")
    print(f"Index file: {index_path}")
    print("=" * 80)

    # Filter components if specified
    if args.components:
        components_config = [c for c in components_config if c['name'] in args.components]
    
    # Separate components and pipelines
    components_only = [c for c in components_config if c.get('type') != 'pipeline']
    pipelines_only = [c for c in components_config if c.get('type') == 'pipeline']
    
    # Filter pipelines if specified
    if args.pipelines:
        pipelines_only = [p for p in pipelines_only if p['name'] in args.pipelines]

    # Track component versions
    component_versions = {}

    # Deploy components first (in dependency order)
    if components_only:
        print("\n" + "=" * 80)
        print("PHASE 1: Deploying Components")
        print("=" * 80)
        
        # Sort components by dependencies
        sorted_component_names = topological_sort(components_only)
        
        for comp_name in sorted_component_names:
            comp = next(c for c in components_only if c['name'] == comp_name)
            
            print(f"\n{'=' * 80}")
            print(f"Component: {comp['name']}")
            print(f"{'=' * 80}")
            
            spec_path = script_dir / comp['path']
            if not spec_path.exists():
                print(f"WARNING: Spec file not found at {spec_path}, skipping", file=sys.stderr)
                continue
            
            # Get current version and bump it
            old_version = get_version_from_spec(spec_path)
            new_version = bump_version(old_version)
            print(f"Version: {old_version} -> {new_version}")
            
            # Update version and registry references
            updated_content = update_version_in_spec(spec_path, new_version)
            if deployment_target == 'registry':
                updated_content = update_registry_references(spec_path, registry_name, component_versions)
                # Re-apply version update after registry update
                tmp_path = spec_path.with_suffix('.tmp')
                write_file(tmp_path, updated_content)
                updated_content = update_version_in_spec(tmp_path, new_version)
                tmp_path.unlink()
            
            if not args.dry_run:
                write_file(spec_path, updated_content)
                print(f"✓ Updated {spec_path}")
            else:
                print(f"[DRY RUN] Would update {spec_path}")
            
            # Track version
            component_versions[comp['name']] = new_version
            
            # Deploy component
            deploy_cmd = [
                'az', 'ml', 'component', 'create',
                '--file', str(spec_path.resolve())
            ]
            
            if deployment_target == 'workspace':
                deploy_cmd.extend([
                    '--resource-group', args.resource_group,
                    '--workspace-name', args.workspace_name
                ])
            else:
                deploy_cmd.extend(['--registry-name', registry_name])
            
            run_command(deploy_cmd, dry_run=args.dry_run)
            
            if not args.dry_run:
                print(f"✓ Deployed {comp['name']} version {new_version}")

    # Deploy pipelines (after all components are deployed)
    if pipelines_only:
        print("\n" + "=" * 80)
        print("PHASE 2: Deploying Pipelines")
        print("=" * 80)
        
        # Sort pipelines by dependencies
        sorted_pipeline_names = topological_sort(pipelines_only)
        
        for pipeline_name in sorted_pipeline_names:
            pipeline = next(p for p in pipelines_only if p['name'] == pipeline_name)
            
            print(f"\n{'=' * 80}")
            print(f"Pipeline: {pipeline['name']}")
            print(f"{'=' * 80}")
            
            spec_path = script_dir / pipeline['path']
            if not spec_path.exists():
                print(f"WARNING: Spec file not found at {spec_path}, skipping", file=sys.stderr)
                continue
            
            # Get current version
            old_version = get_version_from_spec(spec_path)
            new_version = bump_version(old_version)
            print(f"Version: {old_version} -> {new_version}")
            
            # Update pipeline content
            updated_content = read_file(spec_path)
            
            # Update component references for trainers
            if 'trainers' in pipeline:
                for trainer_name in pipeline['trainers']:
                    if trainer_name in component_versions:
                        trainer_version = component_versions[trainer_name]
                        print(f"Updating {trainer_name} reference to version {trainer_version}")
                        updated_content = update_component_reference_in_pipeline(
                            spec_path, trainer_name, trainer_version, registry_name
                        )
                        # Write intermediate result
                        write_file(spec_path, updated_content)
            
            # Update component references for dependencies
            if 'dependencies' in pipeline:
                for dep_name in pipeline['dependencies']:
                    if dep_name in component_versions:
                        dep_version = component_versions[dep_name]
                        print(f"Updating {dep_name} reference to version {dep_version}")
                        updated_content = update_component_reference_in_pipeline(
                            spec_path, dep_name, dep_version, registry_name
                        )
                        write_file(spec_path, updated_content)
            
            # Update all registry references
            if deployment_target == 'registry':
                updated_content = update_registry_references(spec_path, registry_name, component_versions)
                write_file(spec_path, updated_content)
            
            # Update pipeline version
            updated_content = update_version_in_spec(spec_path, new_version)
            
            if not args.dry_run:
                write_file(spec_path, updated_content)
                print(f"✓ Updated {spec_path}")
            else:
                print(f"[DRY RUN] Would update {spec_path}")
            
            # Deploy pipeline
            deploy_cmd = [
                'az', 'ml', 'component', 'create',
                '--file', str(spec_path.resolve())
            ]
            
            if deployment_target == 'workspace':
                deploy_cmd.extend([
                    '--resource-group', args.resource_group,
                    '--workspace-name', args.workspace_name
                ])
            else:
                deploy_cmd.extend(['--registry-name', registry_name])
            
            run_command(deploy_cmd, dry_run=args.dry_run)
            
            if not args.dry_run:
                print(f"✓ Deployed {pipeline['name']} version {new_version}")

    # Summary
    print("\n" + "=" * 80)
    print("Deployment Summary")
    print("=" * 80)
    
    if components_only:
        print("\nComponents deployed:")
        for comp in components_only:
            if comp['name'] in component_versions:
                print(f"  - {comp['name']}: {component_versions[comp['name']]}")
    
    if pipelines_only:
        print("\nPipelines deployed:")
        for pipeline in pipelines_only:
            print(f"  - {pipeline['name']}")
    
    if deployment_target == 'workspace':
        print(f"\nWorkspace: {args.resource_group}/{args.workspace_name}")
    else:
        print(f"\nRegistry: {registry_name}")

    if args.dry_run:
        print("\n[DRY RUN] No changes were made. Run without --dry-run to deploy.")
    else:
        print("\n✓ Deployment completed successfully!")

    print("=" * 80)


if __name__ == '__main__':
    main()
