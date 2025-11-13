#!/usr/bin/env python
"""
Preprocess checkpoint by stripping "base_model.model." prefix from weight_map keys
in model.safetensors.index.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Tuple
import shutil


def preprocess_checkpoint(checkpoint_path: str, output_path: str = None) -> Tuple[bool, str]:
    """
    Preprocess checkpoint by stripping 'base_model.model.' prefix from weight_map keys.
    
    Args:
        checkpoint_path: Path to original checkpoint directory
        output_path: If provided, copy checkpoint to this path with corrections
        
    Returns:
        Tuple of (success, path_to_use) where path_to_use is either output_path or checkpoint_path
    """
    print(f"\n{'='*80}")
    print(f"Preprocessing checkpoint: {checkpoint_path}")
    if output_path:
        print(f"Output path: {output_path}")
    print(f"{'='*80}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint path does not exist: {checkpoint_path}")
        return False, checkpoint_path
    
    # Look for model.safetensors.index.json
    index_file = os.path.join(checkpoint_path, "model.safetensors.index.json")
    
    if not os.path.exists(index_file):
        print(f"ℹ️  No model.safetensors.index.json found at: {index_file}")
        print(f"   Checkpoint may not need preprocessing or uses different format")
        return True, checkpoint_path
    
    print(f"✅ Found index file: {index_file}")
    
    # Read the index file
    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read index file: {e}")
        return False, checkpoint_path
    
    # Check if weight_map exists
    if 'weight_map' not in index_data:
        print(f"ℹ️  No 'weight_map' found in index file, skipping preprocessing")
        return True, checkpoint_path
    
    weight_map = index_data['weight_map']
    print(f"📊 Found {len(weight_map)} entries in weight_map")
    
    # Check if any keys have the prefix
    prefix = "base_model.model."
    keys_with_prefix = [k for k in weight_map.keys() if k.startswith(prefix)]
    
    if not keys_with_prefix:
        print(f"ℹ️  No keys with prefix '{prefix}' found, no preprocessing needed")
        return True, checkpoint_path
    
    print(f"🔧 Found {len(keys_with_prefix)} keys with prefix '{prefix}'")
    print(f"   Example keys before:")
    for key in list(keys_with_prefix)[:3]:
        print(f"     - {key}")
    
    # Determine target directory
    if output_path:
        # Copy checkpoint to output directory
        print(f"📁 Copying checkpoint to: {output_path}")
        try:
            os.makedirs(output_path, exist_ok=True)
            
            # Copy all files from checkpoint to output
            import shutil
            for item in os.listdir(checkpoint_path):
                src = os.path.join(checkpoint_path, item)
                dst = os.path.join(output_path, item)
                
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            
            print(f"✅ Checkpoint copied to output directory")
            target_index_file = os.path.join(output_path, "model.safetensors.index.json")
            target_dir = output_path
        except Exception as e:
            print(f"❌ Failed to copy checkpoint: {e}")
            return False, checkpoint_path
    else:
        # Modify in place - create backup
        backup_file = index_file + ".backup"
        try:
            shutil.copy2(index_file, backup_file)
            print(f"💾 Created backup: {backup_file}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to create backup: {e}")
        
        target_index_file = index_file
        target_dir = checkpoint_path
    
    # Strip the prefix from all keys
    new_weight_map = {}
    for key, value in weight_map.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]  # Remove prefix
            new_weight_map[new_key] = value
        else:
            new_weight_map[key] = value
    
    # Update the index data
    index_data['weight_map'] = new_weight_map
    
    print(f"   Example keys after:")
    for new_key in list(new_weight_map.keys())[:3]:
        print(f"     - {new_key}")
    
    # Write the modified index file
    try:
        with open(target_index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2)
        print(f"✅ Successfully updated index file")
        return True, target_dir
    except Exception as e:
        print(f"❌ Failed to write updated index file: {e}")
        return False, checkpoint_path


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess checkpoint by stripping 'base_model.model.' prefix from weight_map"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to original checkpoint directory"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional: Path to copy corrected checkpoint (if not provided, modifies in-place)"
    )
    
    args = parser.parse_args()
    
    success, path_used = preprocess_checkpoint(args.checkpoint_path, args.output_path)
    
    if success:
        print(f"\n✅ Checkpoint preprocessing completed successfully")
        print(f"   Path to use: {path_used}")
        return 0
    else:
        print(f"\n❌ Checkpoint preprocessing failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
