#!/usr/bin/env python3
"""
Rename expert tensor keys in a safetensors model.

Transforms expert keys from:
  model.layers.{L}.mlp.experts.{down,gate_up}.{E}.{suffix}
To:
  model.layers.{L}.mlp.experts.{E}.{down_proj,gate_up_proj}_{suffix}

For bias, the suffix becomes _bias. For other suffixes (qweight, qzeros,
scales, g_idx), they are preserved as _{suffix}.

Non-expert tensors pass through unchanged.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

EXPERT_RE = re.compile(
    r"^(model\.layers\.\d+\.mlp\.experts)"
    r"\.(down|gate_up)\.(\d+)"
    r"\.(.+)$"
)


def rename_key(key):
    """Rename an expert key, or return None if not an expert key."""
    m = EXPERT_RE.match(key)
    if not m:
        return None
    prefix, proj, eidx, suffix = m.groups()
    return f"{prefix}.{eidx}.{proj}_proj.{suffix}"


def process_shard(tensors):
    """Rename expert keys in one shard, pass through the rest."""
    out = {}
    for key, tensor in tensors.items():
        new_key = rename_key(key)
        out[new_key if new_key is not None else key] = tensor
    return out


def main():
    p = argparse.ArgumentParser(
        description="Rename expert tensor keys in a safetensors model"
    )
    p.add_argument("input", type=Path, help="Input HuggingFace model directory")
    p.add_argument("output", type=Path, help="Output directory")
    p.add_argument(
        "--dry-run", action="store_true", help="Print key mapping without writing"
    )
    args = p.parse_args()

    if not args.input.is_dir():
        print(f"Error: {args.input} is not a directory", file=sys.stderr)
        sys.exit(1)

    safetensors_files = sorted(args.input.glob("*.safetensors"))
    if not safetensors_files:
        print("No safetensors files found", file=sys.stderr)
        sys.exit(1)

    index_path = args.input / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else None

    # Dry-run mode: just print the mapping
    if args.dry_run:
        for f in safetensors_files:
            print(f"\n{f.name}:")
            tensors = load_file(str(f))
            for key in sorted(tensors.keys()):
                new_key = rename_key(key)
                if new_key is not None:
                    print(f"  {key}  ->  {new_key}")
        return

    args.output.mkdir(parents=True, exist_ok=True)

    # Copy non-safetensors files (config.json, tokenizer, etc.)
    for src in sorted(args.input.iterdir()):
        if src.name.startswith("."):
            continue  # skip hidden files/dirs
        if src.name == "model.safetensors.index.json":
            continue  # rebuilt below
        if src.suffix == ".safetensors":
            continue  # handled below
        dst = args.output / src.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        print(f"Copied {src.name}")

    # Process safetensors shards
    new_weight_map = {}

    for f in safetensors_files:
        print(f"Processing {f.name}...")
        tensors = load_file(str(f))
        result = process_shard(tensors)

        out_path = args.output / f.name
        print(f"  Writing {out_path.name} ({len(result)} tensors)")
        save_file(result, str(out_path))

        for key in result:
            new_weight_map[key] = f.name

    # Update index if present
    if index is not None:
        new_index = {
            "metadata": index.get("metadata", {}),
            "weight_map": new_weight_map,
        }
        index_out = args.output / "model.safetensors.index.json"
        index_out.write_text(json.dumps(new_index, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {index_out.name}")

    print("Done.")


if __name__ == "__main__":
    main()
