"""Local VLM reasoner backend placeholder for self-hosted model weights.

Purpose:
    Reserved for Qwen3-VL and related open-weight vision-language models served
    through vLLM or Hugging Face on Kaya GPUs.

Pipeline role:
    Stage M3 will implement this backend behind the `Reasoner` ABC so the
    orchestrator can run local text-only and text+image `ModelInput` payloads
    without importing model-specific code.

Arguments:
    None. This module is import-only until the local backend is implemented.
"""
