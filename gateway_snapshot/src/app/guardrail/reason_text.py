# Wording shared by the scanners' explanations, which become the Reason
# users read on Runtime Events and the endpoint Prompt Details.

from app.guardrail import constants as guardrail_const


def readable_label(raw_label):
    label = guardrail_const.REASON_LABELS.get(raw_label)
    if label:
        return label
    raw_text = str(raw_label)
    # detect-secrets already reports readable names ("AWS Access Key");
    # only ids written as SNAKE_CASE or snake_case need turning into words.
    if "_" not in raw_text and not raw_text.isupper():
        return raw_text
    return raw_text.replace("_", " ").strip().capitalize()


def readable_list(raw_labels):
    labels = []
    for raw_label in raw_labels:
        label = readable_label(raw_label)
        if label not in labels:
            labels.append(label)
    return ", ".join(labels)


def percent(score):
    return f"{score:.0%}"


def language_name(language_code):
    return guardrail_const.LANGUAGE_NAMES.get(language_code, language_code)
