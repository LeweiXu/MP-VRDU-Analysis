"""Checks for complete model closure and parser staging."""

import json

from ops.scripts import prestage


def test_stage_all_discovers_adapter_bases_recursively(tmp_path, monkeypatch):
    snapshots = {}
    for model_id in ("org/adapter", "org/base-adapter", "org/base", "org/vision-config"):
        path = tmp_path / model_id.replace("/", "--")
        path.mkdir()
        snapshots[model_id] = path
    (snapshots["org/adapter"] / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "org/base-adapter"})
    )
    (snapshots["org/base-adapter"] / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "org/base"})
    )
    calls = []

    def fake_snapshot(model_id, *_args, **_kwargs):
        calls.append(model_id)
        return snapshots[model_id]

    monkeypatch.setattr(prestage, "snapshot", fake_snapshot)
    prestage.stage_all(
        ["org/adapter"],
        "retrieval models",
        None,
        tmp_path,
        force=False,
        workers=1,
        include_adapter_bases=True,
        dependency_map={"org/base": ["org/vision-config"]},
    )

    assert calls == ["org/adapter", "org/base-adapter", "org/base", "org/vision-config"]


def test_parser_model_closure_adds_paddle_layout_model():
    raw = {
        "parsers": {"paddleocrvl": "PaddlePaddle/PaddleOCR-VL", "mineru": "org/mineru"},
        "parser_aux_models": {"paddleocrvl": ["PaddlePaddle/PP-DocLayoutV2"]},
    }

    assert prestage.parser_model_closure(raw, ["PaddlePaddle/PaddleOCR-VL"]) == [
        "PaddlePaddle/PaddleOCR-VL",
        "PaddlePaddle/PP-DocLayoutV2",
    ]
    assert prestage.parser_model_closure(raw, ["org/mineru"]) == ["org/mineru"]


def test_prestage_has_no_longdocurl_option():
    args = prestage.build_parser().parse_args([])
    assert not hasattr(args, "skip_longdocurl")


def test_snapshot_directory_match_checks_every_file(tmp_path):
    snapshot = tmp_path / "snapshot"
    dest = tmp_path / "dest"
    (snapshot / "nested").mkdir(parents=True)
    (dest / "nested").mkdir(parents=True)
    (snapshot / "nested" / "weights.bin").write_bytes(b"weights")
    (dest / "nested" / "weights.bin").write_bytes(b"weights")

    assert prestage.snapshot_matches_directory(snapshot, dest)
    (dest / "nested" / "weights.bin").write_bytes(b"bad")
    assert not prestage.snapshot_matches_directory(snapshot, dest)
