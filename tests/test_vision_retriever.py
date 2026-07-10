"""Checks family-specific vision retriever loading options."""

import json
from types import SimpleNamespace

from colpali_engine.models.modernvbert.configuration_modernvbert import ModernVBertConfig

from retrievers.vision import ColModernVbertRetriever, ColQwen25Retriever


def test_colmodernvbert_uses_embedded_text_config(monkeypatch, tmp_path):
    text_config = SimpleNamespace(
        text_model_name="jhu-clsp/ettin-encoder-150m",
        to_dict=lambda: {"model_type": "modernvbert_text", "vocab_size": 50408},
    )
    config = SimpleNamespace(text_config=text_config, freeze_config=None)
    calls = []

    def fake_from_pretrained(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return config

    monkeypatch.setattr(ModernVBertConfig, "from_pretrained", fake_from_pretrained)
    hf_home = tmp_path / "hf"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    retriever = ColModernVbertRetriever(cache_dir=tmp_path / "results")
    kwargs = retriever.model_load_kwargs()

    assert calls == [
        (
            "ModernVBERT/colmodernvbert-base",
            {"cache_dir": hf_home, "local_files_only": True},
        )
    ]
    assert kwargs == {
        "config": config,
        "local_files_only": True,
        "key_mapping": {
            r"^model\.vision_model\.vision_model\.": "model.vision_model.",
            r"^model\.connector\.modality_projection\.weight$": (
                "model.connector.modality_projection.proj.weight"
            ),
            r"^model\.custom_text_proj\.": "custom_text_proj.",
        },
    }
    assert config.freeze_config == {"freeze_text_layers": False}
    assert text_config.text_model_name == str(
        (hf_home / "mpvrdu" / "colmodernvbert-text-config").resolve()
    )
    saved = json.loads((hf_home / "mpvrdu" / "colmodernvbert-text-config" / "config.json").read_text())
    assert saved == {"model_type": "modernbert", "vocab_size": 50408}


def test_other_vision_retrievers_need_no_model_overrides():
    assert ColQwen25Retriever().model_load_kwargs() == {}
