import re
from app.guardrail import constants as guardrail_const
from app.guardrail import redaction
from app.guardrail import schemas
from app.loggers.applicationlogger import app_logger

# Regex patterns for the Malicious Entity scanner
IPV4_PATTERN = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
IPV6_PATTERN = r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|\b(?:[0-9a-fA-F]{1,4}:){1,7}:(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}\b"
IP_PATTERN = rf"(?:{IPV4_PATTERN}|{IPV6_PATTERN})"
URL_PATTERN = r"\b(?:https?://|www\.)[^\s]+\b"
DOMAIN_PATTERN = r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,63}\b"
FILE_HASH_PATTERN = r"\b(?:[a-fA-F0-9]{64}|[a-fA-F0-9]{40}|[a-fA-F0-9]{32})\b"
CVE_PATTERN = r"\bCVE-\d{4}-\d{4,7}\b"

ENTITY_PATTERNS = {
    "ip_address": re.compile(IP_PATTERN, re.IGNORECASE),
    "url": re.compile(URL_PATTERN, re.IGNORECASE),
    "domain": re.compile(DOMAIN_PATTERN, re.IGNORECASE),
    "file_hash": re.compile(FILE_HASH_PATTERN, re.IGNORECASE),
    "cve_id": re.compile(CVE_PATTERN, re.IGNORECASE),
}


class MaliciousEntityScanner:
    name = guardrail_const.MALICIOUS_ENTITY_SCANNER

    def scan(self, prompt, scanner_policy=None, action=None):
        scanner_policy = scanner_policy or guardrail_const.DEFAULT_SCANNER_POLICY
        rule_config = scanner_policy.get("rule_config") or {}
        rules = rule_config.get("rules", [])

        # Build mapping from entity type to its action
        entity_actions = {}
        if rules:
            for rule in rules:
                if not getattr(rule, "get", None) or not rule.get("enabled", True):
                    continue
                entity_type = rule.get("entity")
                action_id = rule.get("action")
                if entity_type in ENTITY_PATTERNS and action_id:
                    entity_actions[entity_type] = action_id
        else:
            # If no rules are configured, default to scanner's configured action for all entities
            default_action = scanner_policy.get("action") or guardrail_const.ACTION_BLOCK
            for entity_type in ENTITY_PATTERNS:
                entity_actions[entity_type] = default_action

        if not entity_actions:
            return schemas.ScannerResult(
                scanner_name=self.name,
                detected=False,
                categories=[],
                confidence=1.0,
                risk_score=0.0,
                modified_prompt=prompt,
                explanation="No valid entity rules configured.",
            )

        matched_entities = []
        matching_categories = []
        for entity_type, entity_action in entity_actions.items():
            if action is not None and entity_action != action:
                continue
            pattern = ENTITY_PATTERNS[entity_type]
            for match in pattern.finditer(prompt):
                matched_text = match.group(0)
                matched_entities.append((entity_type, matched_text))
                if entity_type not in matching_categories:
                    matching_categories.append(entity_type)

        if not matched_entities:
            return schemas.ScannerResult(
                scanner_name=self.name,
                detected=False,
                categories=[],
                confidence=1.0,
                risk_score=0.0,
                modified_prompt=prompt,
                explanation="No malicious entities detected.",
            )

        formatted_matches = [f"[{entity_type}] {matched_text}" for entity_type, matched_text in matched_entities]
        app_logger.info(f"MaliciousEntityScanner matched sub-rules for phase '{action}': {formatted_matches}")

        explanation = f"Detected malicious entities: {', '.join(sorted(set(formatted_matches)))}."

        is_modify_phase = (action == guardrail_const.ACTION_MODIFY)
        should_redact = is_modify_phase or (action is None and any(act == guardrail_const.ACTION_MODIFY for act in entity_actions.values()))

        redacted_prompt = prompt
        sensitive_values = []
        if should_redact:
            redaction_style = scanner_policy.get("redaction_style")
            custom_text = scanner_policy.get("custom_text")
            values_to_redact = [
                matched_text for entity_type, matched_text in matched_entities
                if is_modify_phase or entity_actions.get(entity_type) == guardrail_const.ACTION_MODIFY
            ]
            redacted_values = set()
            for matched_text in sorted(values_to_redact, key=len, reverse=True):
                if matched_text in redacted_values:
                    continue
                redacted_values.add(matched_text)
                replacement = redaction.redact_value(matched_text, redaction_style, custom_text)
                redacted_prompt = redacted_prompt.replace(matched_text, replacement)
            sensitive_values = sorted(redacted_values)
        else:
            sensitive_values = sorted({matched_text for _, matched_text in matched_entities})

        return schemas.ScannerResult(
            scanner_name=self.name,
            detected=True,
            categories=matching_categories or ["malicious_entity"],
            confidence=1.0,
            risk_score=1.0,
            modified_prompt=redacted_prompt if should_redact else prompt,
            sensitive_values=sensitive_values,
            explanation=explanation,
        )
