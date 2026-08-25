"""Reporting and post-hoc analysis over a finished sweep's `runs.jsonl`.

Nothing in the training path imports these; they read `train/results/` and write
tables:

    .venv/bin/python -m train.analysis.report_arms train/results/arms_v3/runs.jsonl
    .venv/bin/python -m train.analysis.project_cost train/results/arms_v3_timing
    .venv/bin/python -m train.analysis.analyse_t17 train/results/arms_v3
    .venv/bin/python -m train.analysis.length_stats
    .venv/bin/python -m train.analysis.rescore
"""
