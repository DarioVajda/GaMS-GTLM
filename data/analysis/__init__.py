"""Measurements over the store and the QA corpus.

Each module is a standalone CLI writing a JSON report to `results/`; none of them
is imported by the pipeline.

    measure_extraction.py       entity linking accuracy of the extractor
    measure_colloc_sampling.py  the collocation cap and its sampler
    measure_ball_mix.py         node-kind mix and size percentiles of the balls
    scan_gender.py              gender coverage per part of speech in the raw KG
    scan_multi_writtenrep.py    forms carrying several writtenRep values
    check_gender_reachability.py  is a noun's gender inside its own ball?
    find_genderless_nouns.py    noun entries with no gender triple
    probe_gender_parse.py       gender parsing on one raw file
    probe_gender_adj.py         form-level vs entry-level gender on adjectives
"""
