"""QA dataset generation for the GTLM lexicographical model.

Design: ../QA_DATASET_DESIGN.md (what the dataset is).
Per-type specification: ../QA_TASKS.md (how each question type is generated).

    sl.py             Slovene collation, the grader's normalization, agreement
    store.py          read-side view of a GTLM store, shaped for QA
    seeds.py          the D8 seed pool, the D9 proxy, D10 bands, the D11 split
    d5b.py            the sense -> kolokacija cap and its seeded sampler
    spec.py           the per-type answer contract (arity, separator, mode)
    templates.py      question frames, with the Tier A withheld set marked
    gen.py            the per-type answer generators
    grade.py          the grader -- a script, no model in the loop
    build_dataset.py  the driver
    selftest.py       the C-checks of QA_TASKS.md section 2
"""
