#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model weights and save the result.

This is the standalone version of the merge path that used to live inline
in ``run_post_train_eval.py``. It's used in two places now:

  1. **SFT -> GRPO handoff.** After ``sft_qwen3_32b.py`` produces a LoRA
     adapter, we merge it into Qwen3-32B so GRPO can train a *fresh*
     adapter on top of SFT-warm-started weights (no adapter stacking).
  2. **Post-train eval in "merged" mode.** Same mechanic, but run on the
     GRPO adapter against the SFT-merged base for max vLLM throughput.

The merge itself is CPU/GPU-indifferent from a correctness standpoint,
but loading a 32B base in BF16 peaks at ~64 GB RAM, so in practice you
want 1xH100 (or a big CPU + disk swap). On an H100 it takes ~10 min
including the save.

Typical invocation:

    python rl_training/scripts/merge_lora.py \\
        --base-model Qwen/Qwen3-32B-Instruct \\
        --adapter rl_training/outputs/qwen3_32b_sft/checkpoint-final \\
        --output-dir /workspace/qwen3_32b_sft_merged

The output dir gets:
  * merged safetensors shards
  * config.json / generation_config.json / tokenizer files
  * a ``merge_info.json`` with base + adapter provenance
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def merge_lora(
    base_model: str,
    adapter_path: str,
    output_dir: str,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = True,
    safe_serialization: bool = True,
) -> str:
    """Merge a LoRA adapter into a base causal-LM and save the result.

    Returns the absolute path of the saved merged model.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(torch_dtype, torch.bfloat16)

    logger.info("Loading base model %s (%s)...", base_model, torch_dtype)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    logger.info("Loading tokenizer from base model...")
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)

    logger.info("Attaching adapter from %s...", adapter_path)
    peft_model = PeftModel.from_pretrained(base, adapter_path)

    logger.info("Merging and unloading (may take several minutes)...")
    merged = peft_model.merge_and_unload()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Writing merged model to %s", out)
    merged.save_pretrained(str(out), safe_serialization=safe_serialization)
    tok.save_pretrained(str(out))

    info = {
        "base_model": base_model,
        "adapter_path": str(Path(adapter_path).resolve()),
        "output_dir": str(out.resolve()),
        "torch_dtype": torch_dtype,
        "safe_serialization": safe_serialization,
    }
    (out / "merge_info.json").write_text(json.dumps(info, indent=2))
    logger.info("Wrote merge_info.json")

    return str(out.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model")
    parser.add_argument("--base-model", required=True,
                        help="HF repo id or local path for the base model")
    parser.add_argument("--adapter", required=True,
                        help="Path to the LoRA adapter directory (contains adapter_config.json)")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write the merged safetensors + tokenizer")
    parser.add_argument("--torch-dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--no-trust-remote-code", action="store_true",
                        help="Disable trust_remote_code for the base model")
    parser.add_argument("--no-safe-serialization", action="store_true",
                        help="Save as pytorch_model.bin instead of safetensors")
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    if not adapter_path.is_dir():
        parser.error(f"Adapter directory not found: {adapter_path}")
    if not (adapter_path / "adapter_config.json").exists():
        parser.error(
            f"{adapter_path}/adapter_config.json is missing. Is this a PEFT adapter?"
        )

    out = merge_lora(
        base_model=args.base_model,
        adapter_path=str(adapter_path),
        output_dir=args.output_dir,
        torch_dtype=args.torch_dtype,
        trust_remote_code=not args.no_trust_remote_code,
        safe_serialization=not args.no_safe_serialization,
    )
    logger.info("Done. Merged model at %s", out)


if __name__ == "__main__":
    main()
