"""How one QA item is spelled for the instruction-tuned backbone.

The backbone is `google/gemma-3-1b-it` — an **instruction-tuned** checkpoint,
whose entire post-training lives inside its own turn format:

    <bos><start_of_turn>user
    {question}<end_of_turn>
    <start_of_turn>model
    ODGOVOR: {answer}<end_of_turn>

Until this module existed the trainer fed it the bare string
`"{question}\\nODGOVOR: {answer}"`.  That is not a stylistic difference, it is
the wrong input, in three separate ways:

  * **no `<bos>`.**  Gemma-3 is trained with one on every sequence
    (`add_bos_token: true`), and `TextGraphDataset.tokenize` passes
    `add_special_tokens=False`, so it was dropped everywhere.
  * **no turn markers.**  The `-it` weights were tuned entirely inside
    `<start_of_turn>…<end_of_turn>`; without them the adapter spends its
    capacity relearning a format the checkpoint already knows.
  * **the wrong terminator.**  Training appended `<eos>` where the checkpoint's
    own answers end with `<end_of_turn>`.

Applying the template is therefore the DEFAULT and the only behaviour — there is
no arm in this experiment for which the old spelling is the right one, and a
flag would only preserve the ability to reproduce a mistake.

Where it is applied differs by stack, and the difference is the point:

  * the **plain-LLM baselines** get it the ordinary way.  Their ball has zero
    graph nodes, so the prompt node *is* the whole sequence and what the model
    reads is exactly what `apply_chat_template` writes.  That is what "what a
    standard LLM does" has to mean if arms 4 and 5 are to be a fair floor.
  * the **GTLM arms** get it on the PROMPT NODE ONLY.  The graph nodes stay
    verbatim KG text (`iztočnica: gora (…)`, `oblika: gore (…)`), because they
    are not dialogue: they are retrieved context the adapter reaches through the
    structural bias, and wrapping each of them in a user turn would assert
    hundreds of turns of a conversation that never happened.

That puts `<bos>` at the start of the prompt node, which on the GTLM stack sits
mid-sequence.  Deliberate: `node_position_mode='reset'` gives every node its own
positions from 0, so the prompt node genuinely *is* a sequence start as the model
sees it — and the alternative, dropping it, would leave the GTLM arms as the only
ones with no `<bos>` anywhere, which is one of the three defects above.

One seam, because the training copy, the generation copy, the label mask and the
evaluator's stop condition all have to agree about where the answer starts and
what ends it.  They drifted apart once already (`\\nODGOVOR:` vs `ODGOVOR:`); the
turn markers add three more places for the same bug to live.
"""
from .config import ANSWER_PREFIX

# What the template writes at the end of the model's turn, and therefore the
# token the supervised span now ends on.  Not `<eos>`: gemma-3-1b-it ends a turn
# with this, and `generation_config.json` lists BOTH as terminators.
TURN_END = "<end_of_turn>"


def require_chat_template(tokenizer):
    """The backbone must be instruction-tuned; refuse loudly if it is not.

    A base checkpoint has no `chat_template`, and `apply_chat_template` would
    raise deep inside the dataset build with a message about jinja.  The arms in
    `train/configs/` all name a `-it` model, so reaching here without a template
    means the model was swapped for one this prompt format does not fit — and
    silently falling back to the old bare spelling would reintroduce exactly the
    bug this module exists to remove.
    """
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"{tokenizer.name_or_path!r} ships no chat template, so the prompt "
            f"cannot be written as a chat turn.  This experiment trains an "
            f"INSTRUCTION-TUNED backbone (google/gemma-3-1b-it); a base model "
            f"needs its own prompt format, chosen deliberately, not this one "
            f"with the turn markers quietly dropped.")
    return tokenizer


def chat_prompt(tokenizer, question, answer, with_answer=True):
    """The prompt node's text: one user turn, one model turn.

    `answer` already carries the `ODGOVOR: ` tag (QA_TASKS.md 0.1), so it is the
    model turn's content verbatim and the marker is never re-added.

    With `with_answer=False` the text is cut at exactly the marker:

        <bos><start_of_turn>user
        {question}<end_of_turn>
        <start_of_turn>model
        ODGOVOR:

    which is the generation prompt — the template's own `add_generation_prompt`
    output plus the tag the answer line always opens with.  It is produced by
    TRUNCATING the full text rather than by a second `apply_chat_template` call,
    so the two dataset copies are guaranteed to share a byte-identical prefix.
    Pass 1 of the evaluator declares an item correct without generating it, on
    the argument that greedy decoding from that prefix would have emitted the
    gold string; if the teacher-forced copy and the generation copy could differ
    by so much as a newline, the argument is false and every number is unsound.

    The trailing newline the template writes after `<end_of_turn>` is stripped,
    so the supervised span ends exactly on the stop token.  Keeping it would make
    pass 1 additionally require the model to predict a newline it never has to
    generate (decoding stops at `<end_of_turn>`), turning correct items into
    pass-1 misses that cost a full generation to re-decide.
    """
    require_chat_template(tokenizer)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question},
         {"role": "assistant", "content": answer}],
        tokenize=False)
    text = text.rstrip("\n")
    if not text.endswith(TURN_END):
        raise ValueError(
            f"the chat template did not end the model turn with {TURN_END!r}: "
            f"…{text[-80:]!r}.  `train/evaluate.py` stops generation on that "
            f"token and the label mask supervises through it, so a template "
            f"that ends some other way needs both updated with it.")
    cut = text.rfind(ANSWER_PREFIX)
    if cut < 0:
        raise ValueError(
            f"no {ANSWER_PREFIX!r} in the chat-wrapped prompt: …{text[-200:]!r}. "
            f"Every answer line opens with the tag (QA_TASKS.md 0.1) and the "
            f"label mask cuts on it.")
    if with_answer:
        return text
    return text[:cut + len(ANSWER_PREFIX)]


def stop_token_ids(tokenizer):
    """Every id that ends the model's turn, the canonical one first.

    `<end_of_turn>` leads because it is what the template writes and what the
    supervised span ends on — it is the token pass 1's "and then it stops" refers
    to.  `<eos>` comes along because gemma-3's own `generation_config` lists both
    and a checkpoint that emits it should stop rather than decode to the cap;
    generation stopping on a token pass 1 did not require is safe (the item is
    then graded on the string it produced), while the reverse would not be.
    """
    ids = []
    turn_end = tokenizer.convert_tokens_to_ids(TURN_END)
    if turn_end is not None and turn_end != tokenizer.unk_token_id:
        ids.append(turn_end)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in ids:
        ids.append(tokenizer.eos_token_id)
    if not ids:
        raise ValueError(
            f"{tokenizer.name_or_path!r} has neither {TURN_END!r} nor an EOS "
            f"token, so nothing tells generation to stop.")
    return ids
