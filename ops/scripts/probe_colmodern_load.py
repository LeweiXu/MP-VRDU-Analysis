"""Load ColModernVBERT directly so constructor failures retain their traceback."""

from pathlib import Path

from retrievers.vision import ColModernVbertRetriever


def main() -> None:
    import colpali_engine.models as models
    import torch

    retriever = ColModernVbertRetriever(cache_dir=Path(".cache"), allow_text_fallback=False)
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda:0",
        **retriever.model_load_kwargs(),
    }
    model = models.ColModernVBert.from_pretrained(retriever.model_id, **kwargs).eval()
    processor = models.ColModernVBertProcessor.from_pretrained(
        retriever.model_id,
        local_files_only=True,
    )
    print(type(model).__name__, type(processor).__name__, model.device)


if __name__ == "__main__":
    main()
