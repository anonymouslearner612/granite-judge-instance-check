import re

from detect_secrets.core import scan as detect_secrets_scan
from detect_secrets.core.plugins.util import get_mapping_from_secret_type_to_class
from detect_secrets.settings import transient_settings

from app.guardrail import constants as guardrail_const
from app.guardrail import reason_text
from app.guardrail import redaction
from app.guardrail import schemas

class CustomPatternSecret:
    # Duck-types detect-secrets' own PotentialSecret just enough for
    # SecretScanner's own scan()/redact_secrets() below (they only ever
    # read .type and .secret_value off of a detected secret).
    def __init__(self, secret_type, secret_value):
        self.type = secret_type
        self.secret_value = secret_value


def secret_values_of(detected_secrets):
    values = set()
    for secret in detected_secrets:
        if secret.secret_value:
            values.add(secret.secret_value)
    return sorted(values)


def enabled_detect_secrets_plugins():
    """Every detect-secrets plugin except the ones excluded as too noisy, in its settings format."""
    plugins = []
    for plugin_class in get_mapping_from_secret_type_to_class().values():
        if plugin_class.__name__ not in guardrail_const.SECRET_SCANNER_EXCLUDED_PLUGINS:
            plugins.append({"name": plugin_class.__name__})
    return plugins


def entity_type_for_secret(secret_type):
    for entity, mapped_types in guardrail_const.SECRET_SUB_RULE_ENTITY_MAPPING.items():
        if secret_type in mapped_types:
            return entity
    return None


def action_for_secret(secret, rule_dict, default_action):
    if secret.type in rule_dict:
        return rule_dict[secret.type]
    entity = entity_type_for_secret(secret.type)
    if entity and entity in rule_dict:
        return rule_dict[entity]
    return default_action


class SecretScanner:
    name = guardrail_const.SECRET_SCANNER

    def __init__(self):
        self.plugins_used = enabled_detect_secrets_plugins()

    def scan(self, prompt, scanner_policy=None, action=None):
        scanner_policy = scanner_policy or guardrail_const.DEFAULT_SCANNER_POLICY
        detected_secrets = self.find_secrets(prompt)

        rule_config = scanner_policy.get("rule_config") or {}
        rules = rule_config.get("rules", [])
        rule_dict = {}
        default_action = None

        if rules:
            for rule in rules:
                if not getattr(rule, "get", None) or not rule.get("enabled", True):
                    continue
                entity = rule.get("entity")
                rule_action = rule.get("action")
                if not entity or not rule_action:
                    continue
                rule_dict[entity] = rule_action
                mapped_types = guardrail_const.SECRET_SUB_RULE_ENTITY_MAPPING.get(entity, [entity])
                for secret_type in mapped_types:
                    rule_dict[secret_type] = rule_action
        else:
            default_action = scanner_policy.get("action") or guardrail_const.ACTION_MODIFY
            for mapped_types in guardrail_const.SECRET_SUB_RULE_ENTITY_MAPPING.values():
                for secret_type in mapped_types:
                    rule_dict[secret_type] = default_action

        # Filter detected secrets down to those whose entity is enabled
        eligible_secrets = [
            secret for secret in detected_secrets
            if action_for_secret(secret, rule_dict, default_action) is not None
        ]

        if not eligible_secrets:
            return schemas.ScannerResult(
                scanner_name=self.name,
                detected=False,
                categories=[],
                confidence=1.0,
                risk_score=0.0,
                modified_prompt=prompt,
                explanation="No secrets detected.",
            )

        if action is not None:
            phase_secrets = [
                secret for secret in eligible_secrets
                if action_for_secret(secret, rule_dict, default_action) == action
            ]
            if not phase_secrets:
                return schemas.ScannerResult(
                    scanner_name=self.name,
                    detected=False,
                    categories=[],
                    confidence=1.0,
                    risk_score=0.0,
                    modified_prompt=prompt,
                    explanation="No matching secrets detected.",
                )

            secret_types = sorted({secret.type for secret in phase_secrets})
            if action == guardrail_const.ACTION_MODIFY:
                redacted_prompt = self.redact_secrets(prompt, phase_secrets, scanner_policy)
            else:
                redacted_prompt = prompt

            return schemas.ScannerResult(
                scanner_name=self.name,
                detected=True,
                categories=secret_types,
                confidence=1.0,
                risk_score=1.0,
                modified_prompt=redacted_prompt,
                sensitive_values=secret_values_of(phase_secrets),
                explanation=f"Credentials detected: {reason_text.readable_list(secret_types)}.",
                redactions=[{
                    "value": value,
                    "replacement": redaction.redact_value(value, scanner_policy.get("redaction_style"),
                                                          scanner_policy.get("custom_text"),
                                                          dummy_value=guardrail_const.SECRET_DUMMY_VALUE),
                } for value in secret_values_of(detected_secrets)] if scanner_policy.get(
                    "include_file_redactions") else [],
            )
            
        # action is None: single-pass execution
        secret_types = sorted({secret.type for secret in eligible_secrets})
        redactable_secrets = [
            secret for secret in eligible_secrets
            if action_for_secret(secret, rule_dict, default_action) == guardrail_const.ACTION_MODIFY
        ]
        redacted_prompt = (
            self.redact_secrets(prompt, redactable_secrets, scanner_policy)
            if redactable_secrets else prompt
        )
        return schemas.ScannerResult(
            scanner_name=self.name,
            detected=True,
            categories=secret_types,
            confidence=1.0,
            risk_score=1.0,
            modified_prompt=redacted_prompt,
            sensitive_values=secret_values_of(eligible_secrets),
            explanation=f"Credentials detected: {reason_text.readable_list(secret_types)}.",
        )

    def find_secrets(self, prompt):
        detected_secrets = self.find_plugin_secrets(prompt)
        detected_secrets += self.find_custom_pattern_secrets(prompt)
        return detected_secrets

    def find_plugin_secrets(self, prompt):
        detected_secrets = []
        with transient_settings({"plugins_used": self.plugins_used}):
            for line in prompt.splitlines():
                detected_secrets.extend(detect_secrets_scan.scan_line(line))
        return detected_secrets

    def find_custom_pattern_secrets(self, prompt):
        detected_secrets = []
        for pattern_type in guardrail_const.SECRET_SCANNER_CUSTOM_PATTERN_TYPES:
            pattern = guardrail_const.CUSTOM_PATTERNS[pattern_type]
            for match in re.finditer(pattern, prompt):
                # A pattern with a capture group marks the secret inside a larger match.
                if match.lastindex and match.group(1):
                    secret_value = match.group(1)
                else:
                    secret_value = match.group(0)
                detected_secrets.append(CustomPatternSecret(pattern_type, secret_value))
        return detected_secrets

    def redact_secrets(self, prompt, detected_secrets, scanner_policy):
        # detect-secrets keeps the raw matched value on secret_value (see
        # PotentialSecret.set_secret) since scan_line() runs against the
        # in-memory prompt, not a saved baseline -- only a baseline load
        # clears it to None. Each unique value is replaced everywhere it
        # appears in the prompt, so the same secret is redacted consistently
        # if it's repeated.
        #
        # Longest value first, which is load-bearing: two detectors often
        # match overlapping spans of the same secret, and replacing the
        # shorter one first destroys the longer match so its own replace
        # silently does nothing. That left the secret body in the prompt --
        # detect-secrets reports just "ghp" for a GitHub token while
        # CUSTOM_PATTERNS reports the whole thing, and redacting "ghp" first
        # produced "REDACTED_ABCDEF...", forwarding the token intact.
        redaction_style = secret_redaction_style(scanner_policy)
        custom_text = scanner_policy.get("custom_text")
        typed_values = {}
        if redaction_style == guardrail_const.REDACTION_STYLE_TYPED:
            typed_values = redaction.typed_placeholders(typed_secrets_in_prompt_order(prompt, detected_secrets))

        secrets_longest_first = sorted(
            detected_secrets,
            key=lambda secret: len(secret.secret_value or ""),
            reverse=True,
        )

        redacted_prompt = prompt
        already_redacted_values = set()
        for secret in secrets_longest_first:
            secret_value = secret.secret_value
            if not secret_value or secret_value in already_redacted_values:
                continue
            already_redacted_values.add(secret_value)
            if secret_value in typed_values:
                replacement = typed_values[secret_value]
            else:
                replacement = redaction.redact_value(secret_value, redaction_style, custom_text, dummy_value=guardrail_const.SECRET_DUMMY_VALUE)
            redacted_prompt = redacted_prompt.replace(secret_value, replacement)
        return redacted_prompt


def secret_redaction_style(scanner_policy):
    rule_config = scanner_policy.get("rule_config") or {}
    return rule_config.get("redaction_style") or scanner_policy.get("redaction_style")


def typed_secrets_in_prompt_order(prompt, detected_secrets):
    values = {secret.secret_value for secret in detected_secrets if secret.secret_value}
    numbered = [
        secret for secret in detected_secrets
        if secret.secret_value and not any(secret.secret_value != other and secret.secret_value in other for other in values)
    ]
    numbered.sort(key=lambda secret: prompt.find(secret.secret_value))
    return [(secret.type, secret.secret_value) for secret in numbered]
