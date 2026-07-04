"""Forward Stage-1 feasibility probes through the Kaya runner.

Purpose:
    Thin wrapper used when invoking `cli.run_probe` on Kaya so the file can
    declare Kaya execution headers while reusing the same probe implementation
    as local runs.

Pipeline role:
    Lets agents run loader/tool/model feasibility probes on the login node or
    submit heavy probes through Kaya mechanics without duplicating probe code.

CLI:
    `python -m kaya.kaya run kaya/run_probe.py -- PROBE [probe-options]`

Arguments:
    All arguments after `--` are forwarded unchanged to `cli.run_probe`; see
    `python -m cli.run_probe --help` for the full list.
"""

# kaya: target=login
# kaya: env=true
# kaya: offline=false

from __future__ import annotations

import sys

from cli.run_probe import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
