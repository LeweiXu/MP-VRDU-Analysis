"""Qwen3-VL-8B-Thinking backend: the reasoning variant, constrained only on its
final line so the judge sees the same artifact as every other model."""

from __future__ import annotations

from models.payload import ModelInput
from models.qwen3vl import Qwen3VLBackend, RenderedPrompt, render_prompt
from schema import Question

# The Thinking variant emits its own reasoning block regardless of instruction.
# It is NOT suppressed: the reasoning is the model's trained behaviour and the
# reason for running it. Only the final line is constrained, so the judge-time
# delimiter extraction (last "Answer:") recovers a terse answer. Appended to
# whatever prompt mode is in effect, so mode semantics are preserved and only
# the output contract is added.
THINKING_FINAL = (
    "After your reasoning, write the final line of your response as exactly:\n"
    "Answer: <your answer>\n"
    "Write nothing after that line."
)


class Qwen3VLThinkingBackend(Qwen3VLBackend):
    """Qwen3-VL-8B-Thinking: inherits loading/generation/telemetry, overrides
    only prompt assembly to add the final-line contract under every mode."""

    prompt_template_version = "qwen3vl-thinking-v1"

    def render(self, question: Question, model_input: ModelInput) -> RenderedPrompt:
        instruction = self.prompt_instruction
        # Empty instruction (mode none) still gets the contract; a non-empty
        # mode keeps its text first so the mode's semantics stay intact.
        combined = f"{instruction.strip()}\n{THINKING_FINAL}" if instruction and instruction.strip() else THINKING_FINAL
        return render_prompt(question, model_input, combined)
