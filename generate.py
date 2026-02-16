#!/usr/bin/env python3
"""Generate tokens from GPT-OSS-20B using HuggingFace transformers.

Usage:
  .venv/bin/python3 generate.py --prompt "Hello, world"
  .venv/bin/python3 generate.py --prompt "Once upon a time" --max-new-tokens 100
  .venv/bin/python3 generate.py --prompt-file my_prompt.txt
"""

import argparse
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from transformers import (
  AutoConfig,
  AutoModelForCausalLM,
  AutoTokenizer,
  TextStreamer,
)

MODEL_DIR = "/opt/positron/weights/huggingface/openai/gpt-oss-20b"
WEIGHTS_DIR = (
  "/opt/positron/weights/huggingface/positron-ai/testing/"
  "quantized-ingest/gpt-oss-20b-tron-ingest-best-bf16"
)


def fix_expert_weights(model, weights_path: str, max_layers: int):
  """Fix expert weight loading for GPT-OSS BF16 format.

  The BF16 safetensors store expert weights as nn.Linear-style keys
  (gate_up_proj.weight, down_proj.weight) with [n_experts, out, in] layout,
  but the model uses bare nn.Parameter (gate_up_proj, down_proj) with
  [n_experts, in, out] layout.
  """
  index_file = os.path.join(weights_path, "model.safetensors.index.json")
  if not os.path.exists(index_file):
    return

  with open(index_file) as f:
    index = json.load(f)

  weight_map = index.get("weight_map", {})
  remap = {}
  for k in weight_map:
    layer_match = re.search(r"layers\.(\d+)", k)
    if layer_match and int(layer_match.group(1)) >= max_layers:
      continue
    if k.endswith(".experts.gate_up_proj.weight"):
      remap[k] = (k[: -len(".weight")], True)
    elif k.endswith(".experts.down_proj.weight"):
      remap[k] = (k[: -len(".weight")], True)

  if not remap:
    return

  print(f"Fixing {len(remap)} expert weight keys", file=sys.stderr)
  for orig_key, (target_key, needs_transpose) in remap.items():
    shard_file = os.path.join(weights_path, weight_map[orig_key])
    with safe_open(shard_file, framework="pt") as f:
      tensor = f.get_tensor(orig_key)
    if needs_transpose and len(tensor.shape) == 3:
      tensor = tensor.transpose(1, 2).contiguous()
    parts = target_key.split(".")
    obj = model
    for part in parts[:-1]:
      obj = getattr(obj, part) if not part.isdigit() else obj[int(part)]
    param_name = parts[-1]
    with torch.no_grad():
      getattr(obj, param_name).copy_(tensor)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--model-dir",
    type=str,
    default=MODEL_DIR,
    help="Base model directory (for tokenizer + model code)",
  )
  parser.add_argument(
    "--weights-dir",
    type=str,
    default=WEIGHTS_DIR,
    help="Directory containing BF16 safetensor weights",
  )
  parser.add_argument(
    "--prompt",
    type=str,
    default=None,
    help="Text prompt for generation",
  )
  parser.add_argument(
    "--prompt-file",
    type=str,
    default=None,
    help="File containing text prompt",
  )
  parser.add_argument(
    "--max-new-tokens",
    type=int,
    default=50,
    help="Maximum number of new tokens to generate",
  )
  parser.add_argument(
    "--max-layers",
    type=int,
    default=24,
    help="Number of transformer layers to load",
  )
  parser.add_argument(
    "--temperature",
    type=float,
    default=0.7,
    help="Sampling temperature (0 = greedy)",
  )
  parser.add_argument(
    "--top-p",
    type=float,
    default=0.9,
    help="Nucleus sampling threshold",
  )
  parser.add_argument(
    "--top-k",
    type=int,
    default=50,
    help="Top-k sampling (0 = disabled)",
  )
  parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Device to run on (cpu, cuda, cuda:0, ...)",
  )
  args = parser.parse_args()

  if args.prompt is None and args.prompt_file is None:
    parser.error("Provide either --prompt or --prompt-file")

  if args.prompt_file:
    with open(args.prompt_file) as f:
      prompt = f.read()
  else:
    prompt = args.prompt

  torch.set_num_threads(os.cpu_count() or 8)
  device = torch.device(args.device)

  print(f"Loading tokenizer from {args.model_dir}", file=sys.stderr)
  tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

  print(
    f"Loading model ({args.max_layers} layers) from {args.weights_dir}",
    file=sys.stderr,
  )
  config = AutoConfig.from_pretrained(args.weights_dir)
  config.num_hidden_layers = args.max_layers

  model = AutoModelForCausalLM.from_pretrained(
    args.weights_dir,
    config=config,
    local_files_only=True,
    dtype=torch.bfloat16,
  )
  fix_expert_weights(model, args.weights_dir, args.max_layers)

  model.to(device)
  model.eval()
  print("Model loaded.", file=sys.stderr)

  input_ids = tokenizer(
    prompt, return_tensors="pt", add_special_tokens=True
  ).input_ids.to(device)

  print(
    f"Prompt: {len(input_ids[0])} tokens, generating up to "
    f"{args.max_new_tokens} new tokens",
    file=sys.stderr,
  )
  print(f"--- prompt ---\n{prompt}\n--- generation ---", file=sys.stderr)

  streamer = TextStreamer(
    tokenizer, skip_prompt=True, skip_special_tokens=True
  )

  gen_kwargs = dict(
    input_ids=input_ids,
    max_new_tokens=args.max_new_tokens,
    streamer=streamer,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
  )

  if args.temperature == 0:
    gen_kwargs["do_sample"] = False
  else:
    gen_kwargs["do_sample"] = True
    gen_kwargs["temperature"] = args.temperature
    gen_kwargs["top_p"] = args.top_p
    if args.top_k > 0:
      gen_kwargs["top_k"] = args.top_k

  with torch.no_grad():
    output_ids = model.generate(**gen_kwargs)

  print(
    f"\n--- done ({len(output_ids[0])} total tokens) ---", file=sys.stderr
  )


if __name__ == "__main__":
  main()
