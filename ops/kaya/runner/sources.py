"""Resolve repo-local runnable files: parse `# kaya:` headers, map local paths to
the remote mirror, build the remote python command, and read a spec's run_tag."""

from __future__ import annotations

from pathlib import Path

from .config import HEADER_PREFIX, LOCAL_ROOT, KayaConfig, RunSettings, parse_bool


def parse_kaya_header(path: Path, *, max_lines: int = 40) -> dict[str, str]:
    """Parse `# kaya: key=value` hints from the top of a Python script."""

    hints: dict[str, str] = {}
    if path.suffix != ".py" or not path.is_file():
        return hints
    for index, line in enumerate(path.read_text(errors="replace").splitlines()):
        if index >= max_lines:
            break
        stripped = line.strip()
        if not stripped.startswith(HEADER_PREFIX):
            continue
        payload = stripped[len(HEADER_PREFIX) :].strip()
        if "=" not in payload:
            continue
        key, value = payload.split("=", 1)
        hints[key.strip().lower().replace("_", "-")] = value.strip()
    return hints


def resolve_run_settings(
    path: Path | None,
    *,
    target_override: str = "auto",
    activate_override: bool | None = None,
    offline_override: bool | None = None,
    default_target: str = "login",
) -> RunSettings:
    """Resolve execution settings from Python headers and CLI overrides."""

    header = parse_kaya_header(path) if path else {}
    target = header.get("target", default_target)
    if target_override != "auto":
        target = target_override
    if target not in {"login", "gpu"}:
        raise ValueError(f"invalid target {target!r}; expected login or gpu")

    activate = True
    if "env" in header:
        activate = parse_bool(header["env"], field="env")
    if activate_override is not None:
        activate = activate_override

    offline = target == "gpu"
    if "offline" in header:
        offline = parse_bool(header["offline"], field="offline")
    if offline_override is not None:
        offline = offline_override

    return RunSettings(target=target, activate_env=activate, offline=offline, job_name=header.get("job-name"))


def local_source_path(value: str) -> Path | None:
    """Return a repo-local source path if `value` names one."""

    path = Path(value)
    if not path.is_absolute():
        path = LOCAL_ROOT / path
    try:
        resolved = path.resolve()
        resolved.relative_to(LOCAL_ROOT)
    except (FileNotFoundError, ValueError):
        return None
    return resolved if resolved.exists() else None


def remote_source_path(config: KayaConfig, local_path: Path) -> str:
    """Map a local repo source file to its path inside the Kaya mirror."""

    rel = local_path.resolve().relative_to(LOCAL_ROOT).as_posix()
    return f"{config.remote_root}/{rel}"


def python_module_name(local_path: Path) -> str | None:
    """Return a repo-local Python module name when the path is importable."""

    if local_path.suffix != ".py":
        return None
    rel = local_path.resolve().relative_to(LOCAL_ROOT).with_suffix("")
    parts = rel.parts
    if not parts or any(not part.isidentifier() for part in parts):
        return None
    return ".".join(parts)


def python_command(config: KayaConfig, local_path: Path, args: list[str]) -> list[str]:
    """Build a remote Python command for a repo-local file."""

    module_name = python_module_name(local_path)
    if module_name:
        return ["python", "-m", module_name, *args]
    return ["python", remote_source_path(config, local_path), *args]


def spec_arg(forwarded: list[str]) -> str | None:
    """Pull the `--spec <file>` (or `--spec=<file>`) value out of forwarded args."""

    for index, token in enumerate(forwarded):
        if token == "--spec" and index + 1 < len(forwarded):
            return forwarded[index + 1]
        if token.startswith("--spec="):
            return token.split("=", 1)[1]
    return None


def spec_job_name(forwarded: list[str]) -> str | None:
    """A SLURM job name from the `--spec` run_tag(s), or None if not a spec run.

    A single-run spec names the job after its run_tag (so `g1-quantization-full` is
    distinguishable in squeue instead of every generate job being `generate`); a
    multi-run spec falls back to the spec file stem. Any parse problem returns None
    so the caller keeps its old default rather than failing the submit.
    """

    spec = spec_arg(forwarded)
    if not spec:
        return None
    path = Path(spec)
    if not path.is_absolute():
        path = LOCAL_ROOT / path
    try:
        from experiments.corpus.yaml_spec import load_yaml_specs

        tags = [s.run_tag for s in load_yaml_specs(path) if s.run_tag]
    except Exception:
        return None
    if not tags:
        return None
    return tags[0] if len(tags) == 1 else path.stem.replace("_", "-")
