from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from openai import OpenAI

from rl_training.data.trajectory import Trajectory

logger = logging.getLogger(__name__)


class OpenAIFineTuner:
    """Wrapper around the OpenAI fine-tuning API."""

    def __init__(self, api_key: str | None = None):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def prepare_training_file(self, trajectories: list[Trajectory], output_path: str | Path) -> Path:
        """Convert trajectories to the OpenAI JSONL chat format and save to disk."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for traj in trajectories:
                f.write(traj.to_openai_jsonl_line() + "\n")
        logger.info("Prepared %d trajectories -> %s", len(trajectories), out)
        return out

    def upload_file(self, path: str | Path) -> str:
        """Upload a JSONL file to OpenAI for fine-tuning.  Returns the file ID."""
        with open(path, "rb") as f:
            file_obj = self.client.files.create(file=f, purpose="fine-tune")
        logger.info("Uploaded file %s -> %s", path, file_obj.id)
        return file_obj.id

    def create_job(
        self,
        training_file_id: str,
        model: str = "gpt-4o-mini",
        n_epochs: int = 3,
        suffix: str = "medagent",
    ) -> str:
        """Submit a fine-tuning job.  Returns the job ID."""
        job = self.client.fine_tuning.jobs.create(
            training_file=training_file_id,
            model=model,
            suffix=suffix,
            hyperparameters={"n_epochs": n_epochs},
        )
        logger.info("Created fine-tuning job %s (model=%s, epochs=%d)", job.id, model, n_epochs)
        return job.id

    def wait_for_completion(self, job_id: str, poll_interval: int = 60) -> str:
        """Poll until a fine-tuning job completes.  Returns the fine-tuned model ID."""
        while True:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            logger.info("Job %s status: %s", job_id, status)

            if status == "succeeded":
                model_id = job.fine_tuned_model
                logger.info("Fine-tuning succeeded: %s", model_id)
                return model_id

            if status in ("failed", "cancelled"):
                raise RuntimeError(f"Fine-tuning job {job_id} {status}: {getattr(job, 'error', 'unknown')}")

            time.sleep(poll_interval)

    def run(
        self,
        trajectories: list[Trajectory],
        base_model: str = "gpt-4o-mini",
        suffix: str = "medagent",
        n_epochs: int = 3,
        output_dir: str = "rl_training/outputs/ft_data",
    ) -> str:
        """End-to-end: prepare file -> upload -> create job -> wait -> return model ID."""
        ts = int(time.time())
        path = self.prepare_training_file(
            trajectories, f"{output_dir}/{suffix}_{ts}.jsonl"
        )
        file_id = self.upload_file(path)
        job_id = self.create_job(file_id, model=base_model, n_epochs=n_epochs, suffix=suffix)
        return self.wait_for_completion(job_id)
