"""Local model policy for evaluation.

Loads a base model + LoRA adapter and runs inference using the same
text-based GET/POST/FINISH protocol that ``MedAgentEnv`` expects, so it
can be used with the existing ``Evaluator``.
"""
from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .base_policy import BasePolicy


def _truncate_text(s: str, max_chars: int) -> str:
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    head = max_chars // 2
    tail = max_chars - head - 40
    return s[:head] + "\n...[truncated; omitted " + str(len(s) - head - tail) + " chars]...\n" + s[-tail:]


class LocalPolicy(BasePolicy):
    """Policy backed by a locally-loaded HuggingFace model (optionally with LoRA)."""

    def __init__(
        self,
        model_path: str,
        base_model: str | None = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        device: str | None = None,
        max_user_message_chars: int = 8000,
        max_context_tokens: int = 6144,
        empty_cache_each_act: bool = False,
    ):
        self.model_id = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_user_message_chars = max_user_message_chars
        self.max_context_tokens = max_context_tokens
        self.empty_cache_each_act = empty_cache_each_act

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

        Long FHIR GET responses are truncated per message and the full prompt
        is capped in tokens to avoid CUDA OOM on consumer GPUs.
        """
        messages = []
        for msg in history:
            role = "assistant" if msg["role"] in ("agent", "assistant") else "user"
            content = msg["content"]
            if role == "user":
                content = _truncate_text(content, self.max_user_message_chars)
            messages.append({"role": role, "content": content})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Left-truncate so we keep the end of the prompt (recent turns + gen prompt).
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_tokens,
            truncation_side="left",
        )
        input_ids = enc["input_ids"].to(self.model.device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(self.model.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
        else:
            gen_kwargs["do_sample"] = False

        model_kw: dict[str, Any] = {"input_ids": input_ids}
        if attn is not None:
            model_kw["attention_mask"] = attn

        with torch.no_grad():
            output_ids = self.model.generate(**model_kw, **gen_kwargs)

        new_tokens = output_ids[0][input_ids.shape[1]:]
        out = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if self.empty_cache_each_act and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out
