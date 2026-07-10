"""Checks for isolated environment setup and its command-line contract."""

from pathlib import Path
from types import SimpleNamespace

from ops.scripts import setup_env


def test_machine_choices_are_deployment_gpu_names():
    action = next(action for action in setup_env.build_parser()._actions if action.dest == "machine")
    assert set(action.choices) == {"V100", "H100"}


def test_package_caches_are_forced_inside_target(tmp_path, monkeypatch):
    for name in ("CONDA_PKGS_DIRS", "PIP_CACHE_DIR", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)

    conda_cache = setup_env.prepare_package_cache_env(tmp_path)

    assert conda_cache == tmp_path / ".cache" / "conda-pkgs"
    assert setup_env.os.environ["CONDA_PKGS_DIRS"] == str(conda_cache)
    assert setup_env.os.environ["PIP_CACHE_DIR"] == str(tmp_path / ".cache" / "pip")
    assert setup_env.os.environ["XDG_CACHE_HOME"] == str(tmp_path / ".cache" / "xdg")


def test_all_continues_after_failure_and_retries_once(tmp_path, monkeypatch):
    config = SimpleNamespace(raw={"python_version": "3.11"}, remote_root=str(tmp_path))
    calls: list[str] = []

    monkeypatch.setattr(setup_env, "load_config", lambda _path: config)
    monkeypatch.setattr(
        setup_env,
        "env_prefix",
        lambda _config, name, local=False: Path(tmp_path) / "envs" / name,
    )

    def fake_build(_config, _machine, name, _python, **_kwargs):
        calls.append(name)
        if name == "parse-mineru":
            raise RuntimeError("dependency failure")

    monkeypatch.setattr(setup_env, "build_env", fake_build)

    assert setup_env.main(["--machine", "V100", "--env", "all"]) == 1
    assert calls.count("parse-mineru") == 2
    assert calls.count("core") == 1
    assert calls.count("parse-unlimited") == 1
    assert calls.count("parse-paddleocrvl") == 1

