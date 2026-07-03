"""Run Stage 1 feasibility probes through the Kaya runner."""

# kaya: target=login
# kaya: env=true
# kaya: offline=false

from __future__ import annotations

import sys

from cli.run_probe import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
