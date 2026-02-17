#!/usr/bin/env python3
"""
Convert BF16 HuggingFace model to the weight layout Tron expects.

MoE expert weights in BF16 HuggingFace models use (E, in, out) layout,
but Tron expects (E, out, in) layout matching the mxfp4 storage convention.
See Note [MoE Weight Transpose in Transformers] in ingest/src/FxTypedFx.hs.

This script:
1. Transposes MoE expert weight tensors from (E, in, out) to (E, out, in)
2. Copies all other tensors unchanged
3. Copies config.json and tokenizer files
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Shard size target: ~5 GB
SHARD_SIZE_BYTES = 5 * 1024 * 1024 * 1024

COPY_FILES = [
  "tokenizer.json",
  "tokenizer_config.json",
  "special_tokens_map.json",
  "merges.txt",
  "vocab.json",
  "generation_config.json",
  "chat_template.jinja",
]


def should_transpose_expert_weight(
  name: str, shape: list[int], config: dict
) -> bool:
  """Check if a weight tensor needs transposition for Tron.

  Expert weight tensors in BF16 models are stored as (E, in, out) but
  Tron expects (E, out, in).
  """
  num_experts = config.get("num_local_experts")
  intermediate_size = config.get("intermediate_size")
  hidden_size = config.get("hidden_size")

  if num_experts is None or intermediate_size is None or hidden_size is None:
    return False

  if len(shape) != 3 or shape[0] != num_experts:
    return False

  # gate_up_proj: (E, hidden, 2*inter) -> (E, 2*inter, hidden)
  if "gate_up_proj" in name and "bias" not in name:
    return shape == [num_experts, hidden_size, 2 * intermediate_size]

  # down_proj: (E, inter, hidden) -> (E, hidden, inter)
  # Transpose is needed even when inter == hidden: HF uses x @ W (in, out)
  # but Tron uses linear(x, W) = x @ W^T expecting (out, in).
  if "down_proj" in name and "bias" not in name and "experts" in name:
    return shape == [num_experts, intermediate_size, hidden_size]

  return False


def tensor_byte_size(t: torch.Tensor) -> int:
  return t.nelement() * t.element_size()


def save_sharded(
  tensors: Dict[str, torch.Tensor], output_dir: Path
) -> None:
  """Save tensors as sharded safetensors with an index file."""
  shards: list[tuple[str, Dict[str, torch.Tensor]]] = []
  current_shard: Dict[str, torch.Tensor] = {}
  current_size = 0
  shard_idx = 1

  for name in sorted(tensors.keys()):
    t = tensors[name]
    t_size = tensor_byte_size(t)
    if current_shard and current_size + t_size > SHARD_SIZE_BYTES:
      shard_name = f"model-{shard_idx:05d}-of-TOTAL.safetensors"
      shards.append((shard_name, current_shard))
      current_shard = {}
      current_size = 0
      shard_idx += 1
    current_shard[name] = t
    current_size += t_size

  if current_shard:
    shard_name = f"model-{shard_idx:05d}-of-TOTAL.safetensors"
    shards.append((shard_name, current_shard))

  total_shards = len(shards)

  final_shards = []
  for shard_name, shard_tensors in shards:
    final_name = shard_name.replace("TOTAL", f"{total_shards:05d}")
    final_shards.append((final_name, shard_tensors))

  weight_map: Dict[str, str] = {}
  total_size = 0
  for shard_name, shard_tensors in final_shards:
    shard_path = output_dir / shard_name
    print(f"  Writing {shard_name} ({len(shard_tensors)} tensors)")
    save_file(shard_tensors, str(shard_path))
    for name, t in shard_tensors.items():
      weight_map[name] = shard_name
      total_size += tensor_byte_size(t)

  index = {
    "metadata": {"total_size": total_size},
    "weight_map": dict(sorted(weight_map.items())),
  }
  index_path = output_dir / "model.safetensors.index.json"
  with open(index_path, "w") as f:
    json.dump(index, f, indent=2)
    f.write("\n")
  print(f"  Index: {len(weight_map)} tensors across {total_shards} shards")


def convert(input_dir: str, output_dir: str) -> None:
  input_dir = Path(input_dir)
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  with open(input_dir / "config.json") as f:
    config = json.load(f)

  # Copy config.json as-is
  shutil.copy2(input_dir / "config.json", output_dir / "config.json")
  print("Copied config.json")

  # Copy tokenizer and other files
  for fname in COPY_FILES:
    src = input_dir / fname
    if src.exists():
      shutil.copy2(src, output_dir / fname)
      print(f"Copied {fname}")

  # Load weight index
  with open(input_dir / "model.safetensors.index.json") as f:
    index = json.load(f)
  weight_map = index["weight_map"]

  # Group weights by file
  files_to_weights: Dict[str, list] = {}
  for weight_name, filename in weight_map.items():
    files_to_weights.setdefault(filename, []).append(weight_name)

  # Process all weights
  output_tensors = {}
  n_transposed = 0

  for filename, weight_names in sorted(files_to_weights.items()):
    filepath = input_dir / filename
    print(f"Processing {filename}...")

    with safe_open(str(filepath), framework="pt") as f:
      for name in weight_names:
        tensor = f.get_tensor(name)
        if should_transpose_expert_weight(name, list(tensor.shape), config):
          tensor = tensor.transpose(1, 2).contiguous()
          n_transposed += 1
          print(f"  {name}: transposed to {list(tensor.shape)}")
        output_tensors[name] = tensor

  print(f"\nTransposed {n_transposed} expert weight tensors")
  print(f"Saving to {output_dir}...")
  save_sharded(output_tensors, output_dir)
  print("Done!")


def main():
  parser = argparse.ArgumentParser(
    description="Convert BF16 HuggingFace model to Tron weight layout"
  )
  parser.add_argument(
    "input_dir",
    type=str,
    help="BF16 model directory",
  )
  parser.add_argument(
    "output_dir",
    type=str,
    help="Output directory for converted model",
  )
  args = parser.parse_args()
  convert(args.input_dir, args.output_dir)


if __name__ == "__main__":
  main()
