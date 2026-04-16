"""Local model policy for evaluation.

Loads a base model + LoRA adapter and runs inference using the same
text-based GET/POST/FINISH protocol that ``MedAgentEnv`` expects, so it
can be used with the existing ``Evaluator``.
"""
from __future__ import annotations

import json
import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .base_policy import BasePolicy


class LocalPolicy(BasePolicy):
    """Policy backed by a locally-loaded HuggingFace model (optionally with LoRA)."""

    def __init__(
        self,
        model_path: str,
        base_model: str | None = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        device: str | None = None,
    ):
        self.model_id = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        # If base_model is provided, load base + LoRA adapter
        if base_model:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=quant_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quant_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def act(self, history: list[dict]) -> str:
        """Generate the next action given the conversation history.

        Maps the MedAgentBench history format (role=agent/user) to the
        model's chat template and generates a response.
        """
        messages = []
        for msg in history:
            role = "assistant" if msg["role"] in ("agent", "assistant") else "user"
            messages.append({"role": role, "content": msg["content"]})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
