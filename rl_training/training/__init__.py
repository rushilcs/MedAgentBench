"""Training utilities (SFT, GRPO, cloud sync).

Import concrete modules explicitly (e.g. ``checkpoint_sync``) so lightweight
callers (RunPod ``launch_runpod.sh`` FHIR snapshot download) do not pull in
``expert_collector`` / server stack dependencies.
"""
