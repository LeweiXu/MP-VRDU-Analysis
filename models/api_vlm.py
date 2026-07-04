"""HTTP API reasoner backend placeholder for hosted comparison models.

Purpose:
    Reserved for OpenAI/Gemini/Anthropic-style backends used for comparison
    runs and the different-family judge path. It documents where HTTP-backed
    reasoners belong without letting pipeline code depend on vendor APIs.

Pipeline role:
    Future registry entries in `models.__init__` will instantiate this backend
    behind the `Reasoner` ABC and consume `ModelInput.to_chat_messages()`.

Arguments:
    None. This module is import-only until the API backend is implemented.
"""
