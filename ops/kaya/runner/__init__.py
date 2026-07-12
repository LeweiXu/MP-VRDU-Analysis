"""Internals of the Kaya runner CLI, split by function.

`ops/kaya/kaya.py` is the entry point (arg parsing + dispatch); each module here
owns one slice: config/dataclasses, remote shell execution, source sync, repo-local
source resolution, sbatch generation, job monitoring, the status report, and the
command handlers.
"""
