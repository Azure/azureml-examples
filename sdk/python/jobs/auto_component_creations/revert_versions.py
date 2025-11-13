#!/usr/bin/env python3
"""
Script to revert version bumps in Azure ML component specs.

This script:
1. Reads the components_creation_index.json file
2. Finds all spec.yaml files
3. Reverts versions to their previous values
4. Optionally reverts to a specific version

Usage:
    # Revert all versions by decrementing (0.0.1 -> 0.0.0)
    python revert_versions.py

    # Preview changes without applying them
    python revert_versions.py --dry-run

    # Revert to a specific version for all components
    python revert_versions.py --version 0.0.0

    # Revert specific components only
    python revert_versions.py --components eagle3_training arl_trainer_component

    # Revert specific pipelines only
    python revert_versions.py --pipelines eagle3_chat_completion_pipeline

Arguments:
    --version: Specific version to set for all components (e.g., 0.0.0)
    --components: List of specific component names to revert
    --pipelines: List of specific pipeline names to revert
    --dry-run: Preview changes without applying them
    --index-file: Path to index JSON file (default: components_creation_index.json)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


def read_file(file_path):
    """Read file content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def write_file(file_path, content):
    """Write content to file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def decrement_version(version_str):
    """
    Decrement the version string.
    Handles versions like '0.0.1.y5' -> '0.0.1.y4' and '0.0.1' -> '0.0.0'

    Args:
        version_str: Current version string

    Returns:
        Decremented version string
    """
    # Match pattern like 0.0.1.y4
    match = re.match(r'^(\d+\.\d+\.\d+\.y)(\d+)$', version_str)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        if number > 0:
            return f"{prefix}{number - 1}"
        else:
            # Can't go below y0, return as-is or error
            print(f"WARNING: Version {version_str} is already at minimum (y0)")
            return version_str

    # Match pattern like 0.0.1
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)$', version_str)
    if match:
        major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if patch > 0:
            return f"{major}.{minor}.{patch - 1}"
        elif minor > 0:
            return f"{major}.{minor - 1}.0"
        elif major > 0:
            return f"{major - 1}.0.0"
        else:
            print(f"WARNING: Version {version_str} is already at minimum (0.0.0)")
            return version_str

    raise ValueError(f"Unsupported version format: {version_str}")


def get_version_from_spec(spec_path):
    """Extract version from spec.yaml file."""
    content = read_file(spec_path)
    version_match = re.search(r'^version:\s*(.+)$', content, re.MULTILINE)
    if not version_match:
        raise ValueError(f"Could not find version in {spec_path}")
    return version_match.group(1).strip()


def update_version_in_spec(spec_path, new_version):
    """Update version in a spec.yaml file and return updated content."""
    content = read_file(spec_path)
    updated_content = re.sub(
        r'^version:\s*.+$',
        f'version: {new_version}',
        content,
        count=1,
        flags=re.MULTILINE
    )
    return updated_content


def main():
    parser = argparse.ArgumentParser(
        description='Revert version bumps in Azure ML component specs'
    )
    parser.add_argument(
        '--version', '--ver',
        type=str,
        help='Specific version to set for all selected components (e.g., 0.0.0)'
    )
    parser.add_argument(
        '--components',
        type=str,
        nargs='+',
        help='List of specific component names to revert'
    )
    parser.add_argument(
        '--pipelines',
        type=str,
        nargs='+',
        help='List of specific pipeline names to revert'
    )
    parser.add_argument(
        '--dry-run', '--dry',
        action='store_true',
        help='Preview changes without applying them'
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

    components_config = index_data.get('components', [])
    
    if not components_config:
        print("ERROR: No components found in index file", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("Azure ML Component Version Revert Script")
    print("=" * 80)
    print(f"Index file: {index_path}")
    if args.version:
        print(f"Target version: {args.version}")
    else:
        print("Mode: Decrement versions")
    print("=" * 80)

    # Filter components if specified
    selected_components = []
    
    if args.components or args.pipelines:
        # Filter based on specified components/pipelines
        for comp in components_config:
            if args.components and comp['name'] in args.components:
                selected_components.append(comp)
            elif args.pipelines and comp['name'] in args.pipelines:
                selected_components.append(comp)
    else:
        # Process all components
        selected_components = components_config

    if not selected_components:
        print("WARNING: No components selected to revert", file=sys.stderr)
        sys.exit(0)

    # Track changes
    changes = []

    # Process each component
    for comp in selected_components:
        print(f"\n{'=' * 80}")
        print(f"Processing: {comp['name']} ({comp.get('type', 'component')})")
        print(f"{'=' * 80}")
        
        spec_path = script_dir / comp['path']
        if not spec_path.exists():
            print(f"WARNING: Spec file not found at {spec_path}, skipping", file=sys.stderr)
            continue
        
        try:
            # Get current version
            old_version = get_version_from_spec(spec_path)
            
            # Determine new version
            if args.version:
                new_version = args.version
            else:
                new_version = decrement_version(old_version)
            
            if old_version == new_version:
                print(f"Version: {old_version} (no change)")
                continue
            
            print(f"Version: {old_version} -> {new_version}")
            
            # Update version
            updated_content = update_version_in_spec(spec_path, new_version)
            
            if not args.dry_run:
                write_file(spec_path, updated_content)
                print(f"✓ Updated {spec_path}")
            else:
                print(f"[DRY RUN] Would update {spec_path}")
            
            # Track change
            changes.append({
                'name': comp['name'],
                'path': comp['path'],
                'old_version': old_version,
                'new_version': new_version
            })
            
        except Exception as e:
            print(f"ERROR processing {comp['name']}: {e}", file=sys.stderr)
            continue

    # Summary
    print("\n" + "=" * 80)
    print("Revert Summary")
    print("=" * 80)
    
    if changes:
        print(f"\nReverted {len(changes)} component(s):")
        for change in changes:
            print(f"  - {change['name']}: {change['old_version']} -> {change['new_version']}")
    else:
        print("\nNo changes made.")
    
    if args.dry_run:
        print("\n[DRY RUN] No changes were applied. Run without --dry-run to revert versions.")
    else:
        print("\n✓ Version revert completed successfully!")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
