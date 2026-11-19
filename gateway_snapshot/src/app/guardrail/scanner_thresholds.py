# The per-control "threshold" an operator sets in the console travels all
# the way into each scanner's policy dict, and nothing used to read it:
# every threshold in the engine was a module constant, so the sensitivity
# dial in the UI moved nothing at all. The three model-backed scanners
# resolve theirs through here instead.
#
# Only those three honour it, and that is not an omission. The rule
# scanners are regex matches with no score to compare a threshold against,
# and Secret/PII carry a per-entity confidence table
# (PRESIDIO_ENTITY_WEIGHTS) where one number cannot mean the same thing for
# a credit card and a person's name.

from app.loggers.applicationlogger import app_logger


def resolve_threshold(scanner_policy, default_threshold, scanner_name):
    """
    The policy's "threshold" when it is a usable score, else the scanner's
    own default.

    A threshold is compared against a model score, so it has to sit in
    (0, 1]. Zero would detect everything, including a label scoring zero,
    and anything above 1 can never be reached -- which would quietly turn
    the scanner off, the one outcome an operator adjusting sensitivity
    would never expect.
    """
    threshold = (scanner_policy or {}).get("threshold")
    if threshold is None:
        return default_threshold

    # bool is an int in Python, and True would otherwise read as 1.0.
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        app_logger.warning(
            f"Guardrail scanner '{scanner_name}' has a non-numeric threshold {threshold!r} -- "
            "ignoring it and using the scanner's own default."
        )
        return default_threshold

    if not 0 < threshold <= 1:
        app_logger.warning(
            f"Guardrail scanner '{scanner_name}' has threshold {threshold}, which is outside (0, 1] -- "
            "ignoring it and using the scanner's own default. Thresholds are model scores, not percentages."
        )
        return default_threshold

    return float(threshold)
