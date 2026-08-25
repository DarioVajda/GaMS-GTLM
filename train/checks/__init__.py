"""Standalone checks that must pass before a sweep is trusted.

Each module here is a CLI, not a library -- nothing in the training path imports
them, and they are run by hand or by `train/slurm/run_preflight.sbatch`:

    .venv/bin/python -m train.checks.check_labels --n 10
    .venv/bin/python -m train.checks.test_left_pad
    .venv/bin/python -m train.checks.test_two_pass --max-items 32
    .venv/bin/python -m train.checks.probe_eval --n 32
"""
