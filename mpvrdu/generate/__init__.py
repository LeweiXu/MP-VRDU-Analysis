"""Stage 3: generation. One input-builder, swappable generators.

All generators implement `Generator.answer(question, images, text)`. The input
packing (image / text / both) is identical across them via `InputBuilder`, so
the only thing that changes between a mock run and a real Kaya run is the model.
"""

from .base import Generator, GeneratorInput, InputBuilder  # noqa: F401
from .mock import MockGenerator  # noqa: F401
from .factory import build_generator  # noqa: F401
