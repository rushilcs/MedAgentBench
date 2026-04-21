"""OpenAI Reinforcement Fine-Tuning launcher (analog of ``openai_finetune.py``).

Thin wrapper around ``client.fine_tuning.jobs.create(method={"type":"reinforcement",...})``
for ``o4-mini-2025-04-16``. Polls the job, streams metrics events, and returns
the final ``ft:...`` model id.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

_TERMINAL = {"succeeded", "failed", "cancelled"}


@dataclass
class RFTJobResult:
    job_id: str
    status: str
    fine_tuned_model: str | None
    base_model: str
    suffix: str
    training_file: str
    validation_file: str
    grader_name: str
    last_metrics: dict[str, Any] = field(default_factory=dict)
    events_tail: list[dict[str, Any]] = field(default_factory=list)


def _optional_hp(value: Any) -> Any:
    """Coerce ``"auto"`` / None through; otherwise leave numeric values."""
    if isinstance(value, str) and value.lower() == "auto":
        return "auto"
    return value


def build_python_grader(source_path: str | Path, *, name: str = "medagent_refsol",
                       image_tag: str = "2025-05-08") -> dict[str, Any]:
    """Load the grader source and build the JSON dict RFT expects."""
    src = Path(source_path).read_text()
    return {
        "type": "python",
        "name": name,
        "image_tag": image_tag,
        "source": src,
    }


class OpenAIRFTLauncher:
    def __init__(self, api_key: str | None = None):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def upload_file(self, path: str | Path) -> str:
        with open(path, "rb") as f:
            obj = self.client.files.create(file=f, purpose="fine-tune")
        logger.info("Uploaded %s -> %s", path, obj.id)
        return obj.id

    def validate_grader(self, grader: dict[str, Any], sample_row: dict[str, Any] | None = None) -> None:
        """Call the alpha grader validate (+ run if a sample row is provided).

        Hard-fails (raises) on:
          * validate endpoint errors (always fatal)
          * smoke-run BadRequest exceptions
          * smoke-run server-side errors visible in metadata, including
            ``unauthorized_error`` (org doesn't have python grader access)
            and ``python_grader_runtime_error`` (deterministic code bug)
        """
        try:
            self.client.fine_tuning.alpha.graders.validate(grader=grader)
            logger.info("Grader validation: OK")
        except Exception as exc:
            logger.error("Grader validation FAILED: %s", exc)
            raise
        if sample_row is None:
            return
        item = {k: v for k, v in sample_row.items() if k != "messages"}
        try:
            resp = self.client.fine_tuning.alpha.graders.run(
                grader=grader,
                model_sample="FINISH([])",
                item=item,
            )
        except Exception as exc:
            # Network/permission errors at the API layer should be fatal.
            logger.error("Grader smoke-run request failed: %s", exc)
            raise
        self._raise_on_grader_metadata_errors(resp)
        reward = getattr(resp, "reward", None)
        logger.info("Grader smoke-run: OK reward=%s", reward)

    @staticmethod
    def _raise_on_grader_metadata_errors(resp: Any) -> None:
        """Inspect ``GraderRunResponse.metadata.errors`` and raise on real errors.

        The ``run`` endpoint returns reward=0.0 with a 200 OK even when the
        sandbox refuses to execute the grader (e.g. ``unauthorized_error``).
        We surface those as hard failures so we don't burn another 6h queue.
        """
        md = getattr(resp, "metadata", None)
        if md is None:
            return
        errors = getattr(md, "errors", None)
        if errors is None:
            return

        py_server = bool(getattr(errors, "python_grader_server_error", False))
        py_server_type = getattr(errors, "python_grader_server_error_type", None)
        if py_server and py_server_type == "unauthorized_error":
            raise RuntimeError(
                "Python grader sandbox returned 'unauthorized_error' — "
                "your OpenAI org is not entitled to run python graders even "
                "though RFT is enabled. Switch to a score_model/string_check "
                "grader (see rl_training/training/openai_rft_score_grader.py) "
                "or contact OpenAI support to enable python graders."
            )
        if py_server:
            raise RuntimeError(
                f"Python grader sandbox server error: type={py_server_type!r}. "
                "Check OpenAI status page; the job will fail in the same way."
            )

        py_runtime = bool(getattr(errors, "python_grader_runtime_error", False))
        if py_runtime:
            details = getattr(errors, "python_grader_runtime_error_details", None)
            raise RuntimeError(
                f"Python grader RUNTIME error (deterministic, will fail in "
                f"training too): {details}"
            )

        for flag in (
            "model_grader_server_error",
            "model_grader_refusal_error",
            "model_grader_parse_error",
            "model_grader_exceeded_max_tokens_error",
        ):
            if bool(getattr(errors, flag, False)):
                detail_attr = flag + "_details"
                detail = getattr(errors, detail_attr, None)
                raise RuntimeError(f"Score-model grader failure ({flag}): {detail}")

        if bool(getattr(errors, "other_error", False)):
            raise RuntimeError("Grader returned 'other_error' on smoke run — see metadata.")

    def create_job(
        self,
        *,
        model: str,
        training_file_id: str,
        validation_file_id: str,
        suffix: str,
        grader: dict[str, Any],
        reasoning_effort: str = "medium",
        n_epochs: Any = "auto",
        compute_multiplier: Any = "auto",
        eval_interval: Any = "auto",
        eval_samples: Any = "auto",
        seed: int | None = 42,
    ) -> str:
        hyperparameters: dict[str, Any] = {
            "reasoning_effort": reasoning_effort,
            "n_epochs": _optional_hp(n_epochs),
            "compute_multiplier": _optional_hp(compute_multiplier),
            "eval_interval": _optional_hp(eval_interval),
            "eval_samples": _optional_hp(eval_samples),
        }
        method = {
            "type": "reinforcement",
            "reinforcement": {
                "grader": grader,
                "hyperparameters": hyperparameters,
            },
        }
        create_kwargs: dict[str, Any] = {
            "model": model,
            "training_file": training_file_id,
            "validation_file": validation_file_id,
            "suffix": suffix,
            "method": method,
        }
        if seed is not None:
            create_kwargs["seed"] = seed
        job = self.client.fine_tuning.jobs.create(**create_kwargs)
        logger.info("Created RFT job %s (model=%s suffix=%s)", job.id, model, suffix)
        return job.id

    def poll(
        self,
        job_id: str,
        *,
        poll_interval: float = 60.0,
        event_limit: int = 50,
    ) -> RFTJobResult:
        """Poll job until terminal; stream metrics events to the logger."""
        seen_event_ids: set[str] = set()
        last_metrics: dict[str, Any] = {}
        tail: list[dict[str, Any]] = []

        while True:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            try:
                events = list(self.client.fine_tuning.jobs.list_events(job_id, limit=event_limit).data)
            except Exception as exc:
                logger.warning("list_events failed: %s", exc)
                events = []

            for ev in reversed(events):  # oldest first
                eid = getattr(ev, "id", None) or f"{getattr(ev, 'created_at', '')}-{getattr(ev, 'type', '')}"
                if eid in seen_event_ids:
                    continue
                seen_event_ids.add(eid)
                etype = getattr(ev, "type", None)
                data = getattr(ev, "data", None)
                msg = getattr(ev, "message", "")
                if etype == "metrics" and isinstance(data, dict):
                    last_metrics = data
                    step = data.get("step")
                    train_r = data.get("train_reward_mean") or data.get("train_mean_reward")
                    valid_r = data.get("valid_reward_mean") or data.get("full_valid_mean_reward")
                    logger.info(
                        "step=%s train_reward=%s valid_reward=%s",
                        step, train_r, valid_r,
                    )
                else:
                    logger.info("event[%s] %s", etype, msg[:240])
                tail.append({
                    "id": eid,
                    "type": etype,
                    "message": msg,
                    "data": data if isinstance(data, dict) else None,
                })

            logger.info(
                "job=%s status=%s est_finish=%s",
                job_id, status, getattr(job, "estimated_finish", "?"),
            )

            if status in _TERMINAL:
                return RFTJobResult(
                    job_id=job_id,
                    status=status,
                    fine_tuned_model=getattr(job, "fine_tuned_model", None),
                    base_model=getattr(job, "model", ""),
                    suffix=(getattr(job, "suffix", "") or ""),
                    training_file=getattr(job, "training_file", ""),
                    validation_file=getattr(job, "validation_file", "") or "",
                    grader_name="",
                    last_metrics=last_metrics,
                    events_tail=tail[-200:],
                )

            time.sleep(poll_interval)

    def run(
        self,
        *,
        train_jsonl: str | Path,
        val_jsonl: str | Path,
        grader: dict[str, Any],
        model: str,
        suffix: str,
        reasoning_effort: str = "medium",
        n_epochs: Any = "auto",
        compute_multiplier: Any = "auto",
        eval_interval: Any = "auto",
        eval_samples: Any = "auto",
        seed: int | None = 42,
        poll_interval: float = 60.0,
        sample_row_for_validation: dict[str, Any] | None = None,
    ) -> RFTJobResult:
        """End-to-end: validate grader → upload files → create job → poll → return."""
        self.validate_grader(grader, sample_row=sample_row_for_validation)
        train_id = self.upload_file(train_jsonl)
        val_id = self.upload_file(val_jsonl)
        job_id = self.create_job(
            model=model,
            training_file_id=train_id,
            validation_file_id=val_id,
            suffix=suffix,
            grader=grader,
            reasoning_effort=reasoning_effort,
            n_epochs=n_epochs,
            compute_multiplier=compute_multiplier,
            eval_interval=eval_interval,
            eval_samples=eval_samples,
            seed=seed,
        )
        result = self.poll(job_id, poll_interval=poll_interval)
        result.grader_name = grader.get("name", "")
        return result


def write_ft_outputs(
    *,
    result: RFTJobResult,
    run_dir: Path,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "rft_job_id.txt").write_text(result.job_id + "\n")
    if result.fine_tuned_model:
        (run_dir / "finetuned_model_id.txt").write_text(result.fine_tuned_model + "\n")
    meta = {
        "fine_tuned_model": result.fine_tuned_model,
        "job_id": result.job_id,
        "status": result.status,
        "base_model": result.base_model,
        "suffix": result.suffix,
        "training_file": result.training_file,
        "validation_file": result.validation_file,
        "grader_name": result.grader_name,
        "last_metrics": result.last_metrics,
    }
    if extra_meta:
        meta.update(extra_meta)
    (run_dir / "finetuned_model_id.json").write_text(json.dumps(meta, indent=2) + "\n")
    (run_dir / "rft_events_tail.json").write_text(json.dumps(result.events_tail, indent=2) + "\n")
    logger.info("Wrote RFT outputs to %s", run_dir)
