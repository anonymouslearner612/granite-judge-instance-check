# PII Scanner -- runs Microsoft Presidio's AnalyzerEngine (NER + built-in
# pattern recognizers), restricted to PII_SCANNER_INCLUDED_ENTITIES, plus a
# second regex layer (CUSTOM_PATTERNS) for shapes Presidio doesn't recognize
# out of the box. The two result sets are merged and deduplicated before
# scoring. Detection text is unicode-normalized and stripped of zero-width/
# control characters first (preprocess_text), so an invisible-character
# trick meant to slip PII past the analyzer's tokenizer doesn't work.

import os
import re
import unicodedata

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer import RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from app.core import settings
from app.guardrail import constants as guardrail_const
from app.guardrail import reason_text
from app.guardrail import redaction
from app.guardrail import schemas
from app.loggers.applicationlogger import app_logger


class PIIScanner:
    name = guardrail_const.PII_SCANNER

    def __init__(self):
        self.model = None

    def ensure_model_loaded(self):
        # Loads lazily on first scan() call and caches on self.model, so the
        # analyzer/anonymizer load at most once per process (PIIScanner is
        # built once and reused -- see guardrail_engine.get_default_scanners)
        # instead of being reloaded per request.
        if self.model is None:
            self.model = self.load_model()
        return self.model

    def load_model(self):
        return {"analyzer": self.build_analyzer(), "anonymizer": AnonymizerEngine()}

    def build_analyzer(self):
        # The spaCy model is not installed in the image as a package; it is
        # read from the mounted model directory when present.
        spacy_model_path = settings.SPACY_MODEL_PATH
        if not os.path.exists(spacy_model_path):
            app_logger.warning(f"spaCy model not found at {spacy_model_path}; using the installed spaCy package")
            return AnalyzerEngine()

        target_model = spacy_model_path
        # If the path points to a python package directory instead of raw model data,
        # find the nested model folder containing config.cfg (e.g. en_core_web_lg-3.8.0)
        if os.path.isdir(spacy_model_path) and not os.path.exists(os.path.join(spacy_model_path, "config.cfg")):
            for entry in os.listdir(spacy_model_path):
                subpath = os.path.join(spacy_model_path, entry)
                if os.path.isdir(subpath) and os.path.exists(os.path.join(subpath, "config.cfg")):
                    target_model = subpath
                    break
            else:
                target_model = "en_core_web_lg"

        nlp_configuration = {
            "nlp_engine_name": guardrail_const.PII_NLP_ENGINE_NAME,
            "models": [{"lang_code": guardrail_const.PII_ANALYZER_LANGUAGE, "model_name": target_model}],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
        app_logger.info(f"spaCy model loaded from {spacy_model_path}")
        return AnalyzerEngine(nlp_engine=nlp_engine)

    def scan(self, prompt, scanner_policy=None, action=None):
        # Detects and produces the redacted text; the engine decides
        # what to do with it based on the action and sub-rule actions.
        scanner_policy = scanner_policy or guardrail_const.DEFAULT_SCANNER_POLICY
        engines = self.ensure_model_loaded()

        normalized_prompt = self.preprocess_text(prompt)
        analyzer_results = engines["analyzer"].analyze(
            text=normalized_prompt,
            language=guardrail_const.PII_ANALYZER_LANGUAGE,
            entities=guardrail_const.PII_SCANNER_INCLUDED_ENTITIES,
        )
        custom_results = self.detect_custom_entities(normalized_prompt, analyzer_results)
        detected_entities = self.deduplicate_entities(analyzer_results + custom_results)
        eligible_entities = self.confident_entities(detected_entities)

        rule_config = scanner_policy.get("rule_config") or {}
        rules = rule_config.get("rules", [])
        rule_dict = {}
        if rules:
            for rule in rules:
                if not getattr(rule, "get", None) or not rule.get("enabled", True):
                    continue
                entity = rule.get("entity")
                rule_action = rule.get("action")
                if not entity or not rule_action:
                    continue
                mapped_types = guardrail_const.PII_SUB_RULE_ENTITY_MAPPING.get(entity, [entity.upper()])
                for presidio_type in mapped_types:
                    rule_dict[presidio_type] = rule_action
        else:
            default_action = scanner_policy.get("action") or guardrail_const.ACTION_MODIFY
            for mapped_types in guardrail_const.PII_SUB_RULE_ENTITY_MAPPING.values():
                for presidio_type in mapped_types:
                    rule_dict[presidio_type] = default_action

        eligible_entities = [entity for entity in eligible_entities if entity.entity_type in rule_dict]

        if action is not None:
            phase_entities = [entity for entity in eligible_entities if rule_dict.get(entity.entity_type) == action]
            if not phase_entities:
                return schemas.ScannerResult(
                    scanner_name=self.name,
                    detected=False,
                    categories=[],
                    confidence=1.0,
                    risk_score=0.0,
                    modified_prompt=normalized_prompt,
                    explanation="No matching PII detected.",
                )

            entity_types = entity_types_of(phase_entities)
            highest_score = max((entity.score for entity in phase_entities), default=1.0)
            if action == guardrail_const.ACTION_MODIFY:
                entities_to_redact = self.remove_subsumed_entities(phase_entities)
                redacted_prompt = self.redact_pii(normalized_prompt, entities_to_redact, scanner_policy, engines["anonymizer"])
                sensitive_values = entity_values_of(normalized_prompt, entities_to_redact)
            else:
                entities_to_redact = []
                redacted_prompt = normalized_prompt
                sensitive_values = entity_values_of(normalized_prompt, self.remove_subsumed_entities(phase_entities))

            return schemas.ScannerResult(
                scanner_name=self.name,
                detected=True,
                categories=entity_types,
                confidence=highest_score,
                risk_score=highest_score,
                modified_prompt=redacted_prompt,
                sensitive_values=sensitive_values,
                explanation=pii_explanation(entity_types),
            )

        if not eligible_entities:
            return schemas.ScannerResult(
                scanner_name=self.name,
                detected=False,
                categories=[],
                confidence=1.0,
                risk_score=0.0,
                explanation="No PII detected.",
            )

        entity_types = entity_types_of(eligible_entities)
        highest_score = max((entity.score for entity in eligible_entities), default=1.0)
        redactable_entities = [
            entity for entity in eligible_entities
            if rule_dict.get(entity.entity_type) == guardrail_const.ACTION_MODIFY
        ]
        entities_to_redact = self.remove_subsumed_entities(redactable_entities)
        redacted_prompt = (
            self.redact_pii(normalized_prompt, entities_to_redact, scanner_policy, engines["anonymizer"])
            if entities_to_redact else normalized_prompt
        )
        return schemas.ScannerResult(
            scanner_name=self.name,
            detected=True,
            categories=entity_types,
            confidence=highest_score,
            risk_score=highest_score,
            modified_prompt=redacted_prompt,
            sensitive_values=entity_values_of(normalized_prompt, to_redact),
            redactions=[{
                "value": normalized_prompt[entity.start:entity.end],
                "replacement": redaction.redact_value(normalized_prompt[entity.start:entity.end],
                    scanner_policy.get("redaction_style"), scanner_policy.get("custom_text"),
                    dummy_value=pii_dummy_value_for_entity_type(entity.entity_type)),
            } for entity in to_redact] if scanner_policy.get("include_file_redactions") else [],
            explanation=pii_explanation(entity_types),
        )

    def preprocess_text(self, prompt):
        normalized = unicodedata.normalize("NFKC", prompt)
        normalized = re.sub(guardrail_const.ZERO_WIDTH_CHARS_PATTERN, "", normalized)
        return re.sub(guardrail_const.CONTROL_CHARS_PATTERN, " ", normalized)

    def detect_custom_entities(self, text, analyzer_results=None):
        # Each pattern carries its own score (CUSTOM_PATTERN_SCORES) rather
        # than a shared constant. A flat score meant every regex hit cleared
        # every tier in min_score_for_entity, so the confidence tiering was
        # dead code for custom patterns -- a vendor-prefixed API key and a
        # bare run of phone-shaped digits were trusted identically.
        results = []
        for entity_type, pattern in guardrail_const.CUSTOM_PATTERNS.items():
            # Credentials belong to the Secret Scanner. Reporting them here too
            # let a PII "modify" action redact a key the tenant set Secrets to
            # only log.
            if entity_type in guardrail_const.SECRET_SCANNER_CUSTOM_PATTERN_TYPES:
                continue
            score = guardrail_const.CUSTOM_PATTERN_SCORES.get(entity_type, guardrail_const.DEFAULT_CUSTOM_PATTERN_SCORE)
            for match in re.finditer(pattern, text):
                # A pattern with a capture group marks the value inside a larger match.
                if match.lastindex and match.group(1):
                    start, end = match.start(1), match.end(1)
                else:
                    start, end = match.start(), match.end()
                if entity_type == "EMAIL_ADDRESS" and any(
                    entity.entity_type == "EMAIL_ADDRESS" and entity.start == start and entity.end <= end
                    and entity.score >= self.min_score_for_entity("EMAIL_ADDRESS")
                    for entity in analyzer_results or []
                ):
                    # Prefer an already recognized address. A sentence such as
                    # "user@example.com. Please reply" must not swallow "Please"
                    # as another OCR-separated domain label.
                    continue
                results.append(RecognizerResult(entity_type=entity_type, start=start, end=end, score=score))
        return results

    def deduplicate_entities(self, entities):
        # Presidio and CUSTOM_PATTERNS can both fire on the same span (e.g.
        # an email matched by both); keep only the higher-scored hit per span.
        best_by_span = {}
        for entity in entities:
            span = (entity.start, entity.end)
            if span not in best_by_span or entity.score > best_by_span[span].score:
                best_by_span[span] = entity
        return list(best_by_span.values())

    def confident_entities(self, entities):
        eligible_entities = []
        for entity in entities:
            if entity.score >= self.min_score_for_entity(entity.entity_type):
                eligible_entities.append(entity)
        return eligible_entities

    def remove_subsumed_entities(self, entities):
        # Drops any entity fully contained inside a longer one already kept,
        # so overlapping detections don't produce broken/duplicated
        # redaction spans when passed to the anonymizer.
        ordered = sorted(entities, key=lambda entity: (entity.start, -(entity.end - entity.start)))
        kept = []
        for entity in ordered:
            if not is_inside_any(entity, kept):
                kept.append(entity)
        return kept

    def min_score_for_entity(self, entity_type):
        # Higher-severity entity types (higher weight) are trusted at a lower
        # model confidence, so a missed critical entity is less likely than a
        # false positive on a trivial one.
        weight = guardrail_const.PRESIDIO_ENTITY_WEIGHTS.get(entity_type.upper(), guardrail_const.DEFAULT_PRESIDIO_ENTITY_WEIGHT)
        for highest_weight_in_tier, min_score in guardrail_const.PII_MIN_SCORE_TIERS:
            if weight <= highest_weight_in_tier:
                return min_score
        return guardrail_const.PII_MIN_SCORE_FALLBACK

    def redact_pii(self, prompt, detected_entities, scanner_policy, anonymizer):
        operators = build_pii_operators(scanner_policy, detected_entities, prompt)
        return anonymizer.anonymize(text=prompt, analyzer_results=detected_entities, operators=operators).text


def pii_explanation(entity_types):
    # Credential patterns are also caught by the Secret Scanner and worded the
    # same way there, so the joined Reason shows them once (join_explanations).
    personal_types = []
    credential_types = []
    for entity_type in entity_types:
        if entity_type in guardrail_const.CREDENTIAL_ENTITY_TYPES:
            credential_types.append(entity_type)
        else:
            personal_types.append(entity_type)
    if not personal_types:
        return f"Credentials detected: {reason_text.readable_list(credential_types)}."
    return f"Personal information detected: {reason_text.readable_list(personal_types)}."


def entity_types_of(entities):
    entity_types = set()
    for entity in entities:
        entity_types.add(entity.entity_type)
    return sorted(entity_types)


def entity_values_of(text, entities):
    values = set()
    for entity in entities:
        value = text[entity.start:entity.end]
        if value.strip():
            values.add(value)
    return sorted(values)


def is_inside_any(entity, kept_entities):
    for existing in kept_entities:
        if existing.start <= entity.start and entity.end <= existing.end:
            return True
    return False


def build_pii_operators(scanner_policy, detected_entities, prompt):
    custom_text = scanner_policy.get("custom_text")
    entity_redactions = pii_entity_redactions(scanner_policy)

    def redaction_for(entity_type):
        return entity_redactions.get(entity_type, (scanner_policy.get("redaction_style"), entity_type))

    typed_values = redaction.typed_placeholders([
        (redaction_for(entity.entity_type)[1], prompt[entity.start:entity.end])
        for entity in sorted(detected_entities, key=lambda entity: entity.start)
        if redaction_for(entity.entity_type)[0] == guardrail_const.REDACTION_STYLE_TYPED
    ])

    operators = {}
    for entity_type in entity_types_of(detected_entities):
        redaction_style = redaction_for(entity_type)[0]
        operators[entity_type] = pii_operator(entity_type, redaction_style, custom_text, typed_values)
    return operators


def pii_operator(entity_type, redaction_style, custom_text, typed_values):
    if redaction_style == guardrail_const.REDACTION_STYLE_DUMMY:
        return OperatorConfig("replace", {"new_value": pii_dummy_value_for_entity_type(entity_type)})
    if redaction_style == guardrail_const.REDACTION_STYLE_TYPED:
        return OperatorConfig(
            "custom",
            {"lambda": lambda original_text: typed_values.get(original_text, guardrail_const.DEFAULT_REDACTION_TEXT)},
        )
    return OperatorConfig(
        "custom",
        {"lambda": lambda original_text: redaction.redact_value(original_text, redaction_style, custom_text)},
    )


def pii_entity_redactions(scanner_policy):
    entity_redactions = {}
    rule_config = scanner_policy.get("rule_config") or {}
    for rule in rule_config.get("rules", []):
        entity = rule.get("entity")
        redaction_style = rule.get("redaction_style")
        if not entity or not redaction_style:
            continue
        for presidio_type in guardrail_const.PII_SUB_RULE_ENTITY_MAPPING.get(entity, [entity.upper()]):
            entity_redactions[presidio_type] = (redaction_style, entity)
    return entity_redactions


def pii_dummy_value_for_entity_type(entity_type):
    return guardrail_const.PII_DUMMY_VALUE_BY_ENTITY_TYPE.get(entity_type, guardrail_const.DEFAULT_PII_DUMMY_VALUE)
