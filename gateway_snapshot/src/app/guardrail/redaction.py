import re

from app.guardrail import constants as guardrail_const


def redact_value(original_value, redaction_style, custom_text=None, dummy_value=None):
    if redaction_style == guardrail_const.REDACTION_STYLE_ASTERISKS:
        return guardrail_const.ASTERISK_REDACTION_TEXT
    if redaction_style == guardrail_const.REDACTION_STYLE_CUSTOM:
        return custom_text if custom_text else guardrail_const.DEFAULT_REDACTION_TEXT
    if redaction_style == guardrail_const.REDACTION_STYLE_PARTIAL:
        return partially_mask_value(original_value)
    if redaction_style == guardrail_const.REDACTION_STYLE_DUMMY:
        return dummy_value if dummy_value else guardrail_const.DEFAULT_REDACTION_TEXT
    return guardrail_const.DEFAULT_REDACTION_TEXT


def typed_label(entity_type):
    return re.sub(r"[^A-Z0-9]+", "_", entity_type.upper()).strip("_") or guardrail_const.DEFAULT_REDACTION_TEXT


def typed_placeholders(labelled_values):
    placeholders = {}
    counts = {}
    for label, value in labelled_values:
        if not value or value in placeholders:
            continue
        label = typed_label(label)
        counts[label] = counts.get(label, 0) + 1
        placeholders[value] = f"[{label}_{counts[label]}]"
    return placeholders


def partially_mask_value(original_value):
    if len(original_value) <= guardrail_const.PARTIAL_REDACTION_VISIBLE_CHARS:
        return "*" * len(original_value)

    visible_suffix = original_value[-guardrail_const.PARTIAL_REDACTION_VISIBLE_CHARS:]
    if re.fullmatch(guardrail_const.CUSTOM_PATTERNS["EMAIL_ADDRESS"], original_value):
        # Display the email suffix without OCR whitespace: ". com" should
        # retain ".com", rather than masking the dot and exposing " com".
        compact_email = re.sub(r"[ \t]+", "", original_value)
        visible_suffix = compact_email[-guardrail_const.PARTIAL_REDACTION_VISIBLE_CHARS:]
    masked_prefix = "*" * (len(original_value) - guardrail_const.PARTIAL_REDACTION_VISIBLE_CHARS)
    return masked_prefix + visible_suffix


def obfuscate_text(text, sensitive_values):
    if not text or not sensitive_values or not isinstance(text, str):
        return text

    for value in sorted(set(sensitive_values), key=len, reverse=True):
        if value:
            text = text.replace(value, partially_mask_value(value))
    return text


def obfuscate_payload(payload, sensitive_values):
    if not sensitive_values:
        return payload
    if isinstance(payload, str):
        return obfuscate_text(payload, sensitive_values)
    if isinstance(payload, dict):
        obfuscated = {}
        for key, value in payload.items():
            obfuscated[key] = obfuscate_payload(value, sensitive_values)
        return obfuscated
    if isinstance(payload, (list, tuple)):
        obfuscated_items = []
        for item in payload:
            obfuscated_items.append(obfuscate_payload(item, sensitive_values))
        return obfuscated_items
    return payload
