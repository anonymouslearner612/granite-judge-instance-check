# Shared by every transformer-backed scanner (prompt injection, toxicity,
# language). Those models accept 512 tokens, and they were all built with
# truncation=True, so anything past that was silently discarded -- padding a
# prompt with filler hid the payload entirely. Splitting the text into
# overlapping windows and classifying each one closes that.

from app.guardrail import constants as guardrail_const


def cap_text_length(text):
    """
    Trims text to MAX_SCAN_TEXT_CHARS, keeping the head AND the tail.

    Head-only truncation was an instruction manual: the whole prompt is
    forwarded to the model whatever happens here, so putting the payload
    past character 200,000 meant no scanner with a length limit ever saw
    it. Keeping both ends costs nothing and removes the "pad, then attack"
    version of that. The middle is still dropped -- see
    select_window_starts for the same problem one level down.
    """
    if len(text) <= guardrail_const.MAX_SCAN_TEXT_CHARS:
        return text

    half_budget = (guardrail_const.MAX_SCAN_TEXT_CHARS - len(guardrail_const.SCAN_TRUNCATION_MARKER)) // 2
    return text[:half_budget] + guardrail_const.SCAN_TRUNCATION_MARKER + text[-half_budget:]


def split_into_chunks(text):
    """
    Overlapping word windows, longest-first-fit. The overlap matters: a
    payload sitting across a window boundary would otherwise be split in
    half and look harmless in both halves.
    """
    words = text.split()
    if len(words) <= guardrail_const.SCAN_CHUNK_WORDS:
        return [text]

    step = guardrail_const.SCAN_CHUNK_WORDS - guardrail_const.SCAN_CHUNK_OVERLAP_WORDS
    window_starts = list(range(0, len(words), step))

    chunks = []
    for start in select_window_starts(window_starts):
        chunks.append(" ".join(words[start:start + guardrail_const.SCAN_CHUNK_WORDS]))
    return chunks


def select_window_starts(window_starts):
    """
    Which windows get classified when a prompt has more of them than
    MAX_SCAN_CHUNKS allows.

    This used to take the first N, which covered about 6,300 words of a
    prompt the input cap lets reach 30,000 -- so "bury it after word 6,300"
    hid a payload from every model-backed scanner, reopening in a new form
    the padding attack chunking was built to close. The windows are spread
    evenly across the whole text instead, first and last always included.

    That is coverage, not a guarantee: a fixed budget over text this long
    leaves gaps between the windows it does classify. What it removes is
    the systematically unscanned *region*.
    """
    if len(window_starts) <= guardrail_const.MAX_SCAN_CHUNKS:
        return window_starts
    if guardrail_const.MAX_SCAN_CHUNKS < 2:
        return window_starts[:guardrail_const.MAX_SCAN_CHUNKS]

    stride = (len(window_starts) - 1) / (guardrail_const.MAX_SCAN_CHUNKS - 1)
    return [window_starts[round(position * stride)] for position in range(guardrail_const.MAX_SCAN_CHUNKS)]


def highest_scoring_classification(classifier, text, score_for_classification):
    """
    Runs `classifier` over every chunk of `text` and returns the single
    classification that `score_for_classification` rates highest -- so one
    suspicious window anywhere in a long prompt decides the outcome, rather
    than whatever happened to land in the first 512 tokens.
    """
    chunks = split_into_chunks(cap_text_length(text))

    best_classification = None
    best_score = -1.0
    for chunk in chunks:
        classification = classifier(chunk)[0]
        score = score_for_classification(classification)
        if score > best_score:
            best_score = score
            best_classification = classification
    return best_classification
