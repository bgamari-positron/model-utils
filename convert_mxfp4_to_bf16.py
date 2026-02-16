#!/usr/bin/env python3
"""
Convert MXFP4 quantized weights to unquantized bfloat16 safetensors.

MXFP4 (Microscaling FP4) format:
- FP4 values use E2M1 format (sign + 2-bit exponent + 1-bit mantissa)
- Two FP4 values are packed per byte (low nibble, high nibble)
- Each block of 32 FP4 values shares one E8M0 scale (8-bit exponent only)
- Dequantization: fp4_to_float(nibble) * 2^(scale - 127)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict

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


def convert_mxfp4_weights(
  input_dir: str, output_path: str, output_dtype: torch.dtype = torch.bfloat16
) -> None:
  """Convert MXFP4 quantized model to unquantized safetensors.

  Args:
    input_dir: Directory containing quantized safetensors and config.json
    output_path: Path for output safetensors file
    output_dtype: Output dtype (default bfloat16)
  """
  input_dir = Path(input_dir)

  # Load config
  with open(input_dir / "config.json") as f:
    config = json.load(f)

  quant_config = config.get("quantization_config", {})
  if quant_config.get("quant_method") != "mxfp4":
    raise ValueError(f"Expected mxfp4 quantization, got {quant_config}")

  # Load weight index
  with open(input_dir / "model.safetensors.index.json") as f:
    index = json.load(f)
  weight_map = index["weight_map"]

  # Group weights by file
  files_to_weights: Dict[str, list] = {}
  for weight_name, filename in weight_map.items():
    if filename not in files_to_weights:
      files_to_weights[filename] = []
    files_to_weights[filename].append(weight_name)

  # Process all weights
  output_tensors = {}
  total_weights = len(weight_map)
  processed = 0

  for filename, weight_names in sorted(files_to_weights.items()):
    filepath = input_dir / filename
    print(f"Processing {filename}...")

    with safe_open(str(filepath), framework="pt") as f:
      # Collect blocks and scales for paired dequantization
      blocks_cache = {}
      scales_cache = {}

      for name in weight_names:
        tensor = f.get_tensor(name)

        if name.endswith("_blocks"):
          # Store for later dequantization
          base_name = name[:-7]  # Remove "_blocks"
          blocks_cache[base_name] = tensor
        elif name.endswith("_scales"):
          base_name = name[:-7]  # Remove "_scales"
          scales_cache[base_name] = tensor
        else:
          # Non-quantized weight, keep as-is or convert dtype
          output_tensors[name] = tensor.to(output_dtype)
          processed += 1
          print(f"  [{processed}/{total_weights}] {name}: {tensor.shape}")

      # Dequantize MXFP4 weights
      for base_name in blocks_cache:
        if base_name not in scales_cache:
          raise ValueError(f"Missing scales for {base_name}")

        blocks = blocks_cache[base_name]
        scales = scales_cache[base_name]

        # Dequantize
        dequantized = dequantize_mxfp4_block(blocks, scales)

        # Transpose gate_up_proj: [experts, out, in] -> [experts, in, out]
        if False and "gate_up_proj" in base_name:
          dequantized = dequantized.transpose(-2, -1).contiguous()

        # Output with .weight suffix instead of _blocks/_scales
        output_name = base_name + ".weight"
        output_tensors[output_name] = dequantized.to(output_dtype)

        processed += 2  # blocks + scales -> 1 weight
        print(
          f"  [{processed}/{total_weights}] {output_name}: "
          f"{blocks.shape} + {scales.shape} -> {dequantized.shape}"
        )

  # Save output
  print(f"\nSaving to {output_path}...")
  save_file(output_tensors, output_path)
  print("Done!")

  # Print size comparison
  input_size = sum(
    (input_dir / fn).stat().st_size for fn in files_to_weights
  )
  output_size = Path(output_path).stat().st_size
  print(f"\nInput size:  {input_size / 1e9:.2f} GB")
  print(f"Output size: {output_size / 1e9:.2f} GB")


def main():
  parser = argparse.ArgumentParser(
    description="Convert MXFP4 weights to unquantized safetensors"
  )
  parser.add_argument(
    "input_dir",
    type=str,
    help="Directory containing MXFP4 quantized model",
  )
  parser.add_argument(
    "output_path",
    type=str,
    help="Output safetensors file path",
  )
  parser.add_argument(
    "--dtype",
    type=str,
    default="bfloat16",
    choices=["bfloat16", "float16", "float32"],
    help="Output dtype (default: bfloat16)",
  )
  args = parser.parse_args()

  dtype_map = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
  }

  convert_mxfp4_weights(args.input_dir, args.output_path, dtype_map[args.dtype])


if __name__ == "__main__":
  main()
