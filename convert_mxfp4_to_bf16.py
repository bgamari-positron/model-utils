#!/usr/bin/env python3
"""
Convert MXFP4 quantized weights to unquantized bfloat16 safetensors.

MXFP4 (Microscaling FP4) format:
- FP4 values use E2M1 format (sign + 2-bit exponent + 1-bit mantissa)
- Two FP4 values are packed per byte (low nibble, high nibble)
- Each block of 32 FP4 values shares one E8M0 scale (8-bit exponent only)
- Dequantization: fp4_to_float(nibble) * 2^(scale - 127)

Output is a complete HF model directory loadable via
AutoModelForCausalLM.from_pretrained().
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, NamedTuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file


# FP4 E2M1 lookup table
# Format: sign(1) | exponent(2) | mantissa(1)
# Exponent bias = 1
# Special: exp=0 is subnormal (implicit 0.xxx), exp=1,2,3 are normal (implicit 1.xxx)
FP4_E2M1_TABLE = torch.tensor([
  # exp=0 (subnormal): value = 0.mantissa * 2^(1-bias) = 0.mantissa * 2^0 = 0.mantissa
  0.0,    # 0b0000: +0.0
  0.5,    # 0b0001: +0.5
  # exp=1: value = 1.mantissa * 2^(1-1) = 1.mantissa * 1
  1.0,    # 0b0010: +1.0
  1.5,    # 0b0011: +1.5
  # exp=2: value = 1.mantissa * 2^(2-1) = 1.mantissa * 2
  2.0,    # 0b0100: +2.0
  3.0,    # 0b0101: +3.0
  # exp=3: value = 1.mantissa * 2^(3-1) = 1.mantissa * 4
  4.0,    # 0b0110: +4.0
  6.0,    # 0b0111: +6.0
  # Negative values (sign bit = 1)
  -0.0,   # 0b1000: -0.0
  -0.5,   # 0b1001: -0.5
  -1.0,   # 0b1010: -1.0
  -1.5,   # 0b1011: -1.5
  -2.0,   # 0b1100: -2.0
  -3.0,   # 0b1101: -3.0
  -4.0,   # 0b1110: -4.0
  -6.0,   # 0b1111: -6.0
], dtype=torch.float32)

# Shard size target: ~5 GB
SHARD_SIZE_BYTES = 5 * 1024 * 1024 * 1024

# Tokenizer and config files to copy from the source model directory
COPY_FILES = [
  "tokenizer.json",
  "tokenizer_config.json",
  "special_tokens_map.json",
  "merges.txt",
  "vocab.json",
  "generation_config.json",
]


def e8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
  """Convert E8M0 scale to float multiplier.

  E8M0 is an 8-bit format with only exponent (no mantissa).
  Value = 2^(scale - 127)
  """
  # Cast to int32 to avoid overflow in subtraction
  exp = scale.to(torch.int32) - 127
  return torch.pow(2.0, exp.float())


def dequantize_mxfp4_block(
  blocks: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
  """Dequantize MXFP4 blocks to float.

  Args:
    blocks: uint8 tensor of shape [..., num_blocks, 16]
            Each 16 bytes contains 32 FP4 values (2 per byte)
    scales: uint8 tensor of shape [..., num_blocks]
            E8M0 scale for each block

  Returns:
    float tensor of shape [..., num_blocks * 32]
  """
  # Extract low and high nibbles
  low_nibbles = blocks & 0x0F
  high_nibbles = (blocks >> 4) & 0x0F

  # Interleave: [b0_lo, b0_hi, b1_lo, b1_hi, ...]
  # Shape: [..., num_blocks, 16] -> [..., num_blocks, 32]
  values = torch.stack([low_nibbles, high_nibbles], dim=-1)
  values = values.view(*blocks.shape[:-1], 32)  # [..., num_blocks, 32]

  # Look up FP4 values
  fp4_values = FP4_E2M1_TABLE[values.long()]

  # Apply scales: [..., num_blocks, 1] broadcasts to [..., num_blocks, 32]
  scale_multipliers = e8m0_to_float(scales).unsqueeze(-1)
  dequantized = fp4_values * scale_multipliers

  # Flatten the block dimension
  # [..., num_blocks, 32] -> [..., num_blocks * 32]
  output_shape = list(dequantized.shape[:-2]) + [-1]
  return dequantized.view(*output_shape)


def tensor_byte_size(t: torch.Tensor) -> int:
  return t.nelement() * t.element_size()


def _dtype_size(dtype: torch.dtype) -> int:
  return torch.tensor([], dtype=dtype).element_size()


class TensorInfo(NamedTuple):
  source_file: str
  source_keys: list
  is_quantized: bool
  output_size: int
  expert_index: int = -1  # -1 for non-expert, 0..N-1 for unstacked expert


def is_stacked_expert(name: str) -> bool:
  """Check if tensor name is a stacked expert tensor (no per-expert index)."""
  parts = name.split('.')
  try:
    idx = parts.index('experts')
    return idx + 1 < len(parts) and not parts[idx + 1].isdigit()
  except ValueError:
    return False


def unstacked_expert_name(stacked_name: str, expert_idx: int) -> str:
  """Convert stacked name to per-expert name.

  e.g. model.layers.0.mlp.experts.gate_up_proj
    -> model.layers.0.mlp.experts.0.gate_up_proj
  """
  parts = stacked_name.split('.')
  idx = parts.index('experts')
  parts.insert(idx + 1, str(expert_idx))
  return '.'.join(parts)


def expand_stacked_experts(
  output_info: Dict[str, TensorInfo], num_experts: int
) -> Dict[str, TensorInfo]:
  """Expand stacked expert tensors into per-expert tensors."""
  expanded: Dict[str, TensorInfo] = {}
  for name, info in output_info.items():
    if is_stacked_expert(name):
      per_expert_size = info.output_size // num_experts
      for e in range(num_experts):
        ename = unstacked_expert_name(name, e)
        expanded[ename] = TensorInfo(
          source_file=info.source_file,
          source_keys=info.source_keys,
          is_quantized=info.is_quantized,
          output_size=per_expert_size,
          expert_index=e,
        )
    else:
      expanded[name] = info
  return expanded


def scan_weights(
  input_dir: Path, weight_map: Dict[str, str], output_dtype: torch.dtype
) -> Dict[str, TensorInfo]:
  """Scan weight files to build output tensor metadata without loading data.

  Uses safetensors get_slice to read shapes without loading tensor data.
  Returns a mapping from output tensor name to its metadata.
  """
  files_to_weights: Dict[str, list] = {}
  for weight_name, filename in weight_map.items():
    files_to_weights.setdefault(filename, []).append(weight_name)

  elem_size = _dtype_size(output_dtype)
  output_info: Dict[str, TensorInfo] = {}

  for filename, weight_names in sorted(files_to_weights.items()):
    filepath = input_dir / filename
    with safe_open(str(filepath), framework="pt") as f:
      blocks_info: Dict[str, tuple] = {}
      scales_names: set = set()

      for name in weight_names:
        shape = f.get_slice(name).get_shape()

        if name.endswith("_blocks"):
          base_name = name[:-7]
          # blocks: [..., num_blocks, 16] -> output: [..., num_blocks * 32]
          # Each byte holds 2 FP4 values, so output elements = input bytes * 2
          output_elements = 1
          for d in shape:
            output_elements *= d
          output_elements *= 2
          blocks_info[base_name] = (filename, output_elements * elem_size)

        elif name.endswith("_scales"):
          base_name = name[:-7]
          scales_names.add(base_name)

        else:
          output_elements = 1
          for d in shape:
            output_elements *= d
          output_info[name] = TensorInfo(
            source_file=filename,
            source_keys=[name],
            is_quantized=False,
            output_size=output_elements * elem_size,
          )

      for base_name, (src_file, out_size) in blocks_info.items():
        if base_name not in scales_names:
          raise ValueError(f"Missing scales for {base_name}")
        output_info[base_name] = TensorInfo(
          source_file=src_file,
          source_keys=[base_name + "_blocks", base_name + "_scales"],
          is_quantized=True,
          output_size=out_size,
        )

  return output_info


def plan_shards(
  output_info: Dict[str, TensorInfo],
) -> List[List[str]]:
  """Assign output tensors to shards in sorted name order."""
  shards: List[List[str]] = []
  current_shard: List[str] = []
  current_size = 0

  for name in sorted(output_info.keys()):
    t_size = output_info[name].output_size
    if current_shard and current_size + t_size > SHARD_SIZE_BYTES:
      shards.append(current_shard)
      current_shard = []
      current_size = 0
    current_shard.append(name)
    current_size += t_size

  if current_shard:
    shards.append(current_shard)

  return shards


def convert_mxfp4_weights(
  input_dir: str,
  output_dir: str,
  output_dtype: torch.dtype = torch.bfloat16,
  unstack_experts: bool = False,
) -> None:
  """Convert MXFP4 quantized model to a complete HF model directory.

  Processes one output shard at a time to keep peak memory at ~1 shard
  rather than the entire model.

  Args:
    input_dir: Directory containing quantized safetensors and config.json
    output_dir: Output directory (will be created)
    output_dtype: Output dtype (default bfloat16)
    unstack_experts: Split stacked expert tensors into per-expert tensors
  """
  input_dir = Path(input_dir)
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  # Load config
  with open(input_dir / "config.json") as f:
    config = json.load(f)

  quant_config = config.get("quantization_config", {})
  if quant_config.get("quant_method") != "mxfp4":
    raise ValueError(f"Expected mxfp4 quantization, got {quant_config}")

  # Write config.json without quantization_config
  out_config = {k: v for k, v in config.items() if k != "quantization_config"}
  dtype_name = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
  }[output_dtype]
  out_config["torch_dtype"] = dtype_name
  with open(output_dir / "config.json", "w") as f:
    json.dump(out_config, f, indent=2)
    f.write("\n")
  print(f"Wrote config.json (torch_dtype={dtype_name})")

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

  # Phase 1: Scan to build tensor metadata (shapes only, no data loaded)
  print("\nScanning weights...")
  output_info = scan_weights(input_dir, weight_map, output_dtype)

  if unstack_experts:
    num_experts = config.get("num_local_experts", 0)
    if num_experts > 0:
      output_info = expand_stacked_experts(output_info, num_experts)
      print(f"Unstacking experts ({num_experts} experts per tensor)")

  total_tensors = len(output_info)
  print(f"Found {total_tensors} output tensors")

  # Phase 2: Plan shards
  shard_plan = plan_shards(output_info)
  total_shards = len(shard_plan)
  print(f"Planned {total_shards} output shards\n")

  # Phase 3: Process and write one shard at a time
  weight_map_out: Dict[str, str] = {}
  total_size = 0
  processed = 0

  for shard_idx, tensor_names in enumerate(shard_plan, 1):
    shard_name = f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"
    shard_tensors: Dict[str, torch.Tensor] = {}

    # Group by source file to minimize file opens
    by_source: Dict[str, List[str]] = {}
    for name in tensor_names:
      info = output_info[name]
      by_source.setdefault(info.source_file, []).append(name)

    for source_file, names in by_source.items():
      filepath = input_dir / source_file
      stacked_cache: Dict[tuple, torch.Tensor] = {}
      with safe_open(str(filepath), framework="pt") as f:
        for name in names:
          info = output_info[name]
          if info.expert_index >= 0:
            # Load stacked tensor once and cache, then slice per expert
            cache_key = tuple(info.source_keys)
            if cache_key not in stacked_cache:
              if info.is_quantized:
                blocks = f.get_tensor(info.source_keys[0])
                scales = f.get_tensor(info.source_keys[1])
                stacked = dequantize_mxfp4_block(blocks, scales)
                stacked_cache[cache_key] = stacked.to(output_dtype)
                del blocks, scales, stacked
              else:
                stacked_cache[cache_key] = f.get_tensor(
                  info.source_keys[0]
                ).to(output_dtype)
            shard_tensors[name] = stacked_cache[cache_key][
              info.expert_index
            ].contiguous()
          elif info.is_quantized:
            blocks = f.get_tensor(name + "_blocks")
            scales = f.get_tensor(name + "_scales")
            dequantized = dequantize_mxfp4_block(blocks, scales)
            shard_tensors[name] = dequantized.to(output_dtype)
            del blocks, scales, dequantized
          else:
            tensor = f.get_tensor(info.source_keys[0])
            shard_tensors[name] = tensor.to(output_dtype)
            del tensor

          processed += 1
          print(f"  [{processed}/{total_tensors}] {name}")
      del stacked_cache

    # Write shard and free memory
    shard_path = output_dir / shard_name
    print(f"  Writing {shard_name} ({len(shard_tensors)} tensors)")
    save_file(shard_tensors, str(shard_path))

    for name, t in shard_tensors.items():
      weight_map_out[name] = shard_name
      total_size += tensor_byte_size(t)

    del shard_tensors

  # Write index
  index_out = {
    "metadata": {"total_size": total_size},
    "weight_map": dict(sorted(weight_map_out.items())),
  }
  index_path = output_dir / "model.safetensors.index.json"
  with open(index_path, "w") as f:
    json.dump(index_out, f, indent=2)
    f.write("\n")
  print(f"  Index: {len(weight_map_out)} tensors across {total_shards} shards")

  # Print size comparison
  all_source_files = set(weight_map.values())
  input_size = sum(
    (input_dir / fn).stat().st_size for fn in all_source_files
  )
  output_size = sum(
    f.stat().st_size
    for f in output_dir.iterdir()
    if f.suffix == ".safetensors"
  )
  print(f"\nInput size:  {input_size / 1e9:.2f} GB")
  print(f"Output size: {output_size / 1e9:.2f} GB")
  print("Done!")


def main():
  parser = argparse.ArgumentParser(
    description="Convert MXFP4 weights to unquantized HF model directory"
  )
  parser.add_argument(
    "input_dir",
    type=str,
    help="Directory containing MXFP4 quantized model",
  )
  parser.add_argument(
    "output_dir",
    type=str,
    help="Output directory for the converted model",
  )
  parser.add_argument(
    "--dtype",
    type=str,
    default="bfloat16",
    choices=["bfloat16", "float16", "float32"],
    help="Output dtype (default: bfloat16)",
  )
  parser.add_argument(
    "--unstack-experts",
    action="store_true",
    help="Split stacked expert tensors into per-expert tensors",
  )
  args = parser.parse_args()

  dtype_map = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
  }

  convert_mxfp4_weights(
    args.input_dir, args.output_dir, dtype_map[args.dtype],
    unstack_experts=args.unstack_experts,
  )


if __name__ == "__main__":
  main()
