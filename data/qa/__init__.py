"""QA dataset generation for the GTLM lexicographical model.

Design: ../QA_DATASET_DESIGN.md (what the dataset is).
Per-type specification: ../QA_TASKS.md (how each question type is generated).

    sl.py             Slovene collation, the grader's normalization, agreement
    store.py          read-side view of a GTLM store, shaped for QA
    seeds.py          the D8 seed pool, the D9 proxy, D10 bands, the D11 split
    unlisted.py       real words the lookup cannot reach -- negative flavour (a)
    colloc_sampling.py  the sense -> kolokacija cap and its seeded sampler
    spec.py           the per-type answer contract (mode, key set, the budget)
    pairs.py          the labelled-pair rule -- how a type's cells get labelled
    templates.py      question frames, with the Tier A withheld set marked
    gen.py            the per-type answer generators
    grade.py          the grader -- a script, no model in the loop

The five pipeline stages, in order (see ../README.md):

    build_dataset.py    1  the items
    relabel.py          3  anchors from a real extraction run
    build_balls.py      4  the balls, and the items re-verbalised against them
    build_variants.py   5  the serialised and no-retrieval baselines

and the checks over what they produce:

    selftest.py       the C-checks of QA_TASKS.md section 2
    check_balls.py    ball/dataset agreement and the membership allow-lists
    check_labels.py   C25 -- is every answer label readable off the ball?
    check_variants.py the baselines carry the same items and answers
    inspect.py        read a generated dataset by eye

and one migration, for a corpus generated before the labelled-pair rule:

    migrate_pairs.py  positional lines -> labelled pairs, via the same
                      `pairs.py` the generator now uses
"""
