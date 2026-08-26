# Extractor prompts

`extractor_prompt.txt` is the prompt the D3 entity extractor runs with. It is the
default of `analysis/measure_extraction.py --prompt` and what
`analysis/extract_in_container.sh` passes, so a re-run reproduces the corpus.

`archive/` holds the superseded revisions. They are kept because
`results/extraction_v1.json` and the 1,520-item comparison runs behind it were
measured with them, and a number is not reproducible without the prompt that
produced it.

| file | resolved | used by |
|---|---|---|
| `extractor_prompt.txt` | 95.5 % | the extraction run behind `generated/v2_clean` (job 130298) |
| `archive/extractor_prompt_v2.txt` | — | job 130286, a 1,520-item comparison |
| `archive/extractor_prompt_v1.txt` | — | jobs 130175 and 130285, `results/extraction_v1.json` |

The three differ in how tightly they constrain the output: v1 admits quoted
targets and asks for an empty list on whole-sentence questions, v2 adds typo
tolerance, and the current one adds the metalanguage exclusion list and the
single-word rule for questions that quote a whole sentence.
