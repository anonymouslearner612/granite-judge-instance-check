# Prompt Injection Scanner -- classifies the prompt with the model named by
# PROMPT_INJECTION_MODEL_NAME in constants.py, a binary SAFE/INJECTION
# sequence classifier, run locally via transformers. The classifier pipeline
# loads lazily on first scan() call via ensure_model_loaded() and is cached
# on self.model for the rest of the process.
#
# Switching models is a one-line change in constants.py -- the label and
# threshold that go with each model live there too.

from transformers import pipeline

from app.guardrail import constants as guardrail_const
from app.guardrail import scanner_thresholds
from app.guardrail import schemas
from app.guardrail import text_chunking


def injection_score(classification):
    # Ranks chunks by how injection-like they are, so the most suspicious
    # window in a long prompt is the one that decides the verdict. A SAFE
    # label scores 0 no matter how confident it is -- confidence in
    # "harmless" is not evidence of anything to rank on.
    if classification["label"] != guardrail_const.PROMPT_INJECTION_LABEL:
        return 0.0
    return classification["score"]


class PromptInjectionScanner:
    name = guardrail_const.PROMPT_INJECTION_SCANNER

    def __init__(self):
        self.model = None

    def ensure_model_loaded(self):
        if self.model is None:
            self.model = self.load_model()
        return self.model

    def load_model(self):
        return pipeline("text-classification", model=guardrail_const.PROMPT_INJECTION_MODEL_NAME, truncation=True, max_length=guardrail_const.CLASSIFIER_MAX_TOKENS)

    def scan(self, prompt, scanner_policy=None):
        # The policy's only say here is its threshold: this scanner cannot
        # redact, and the action that decides what a detection means is
        # applied by the engine, not here.
        if len(prompt.split()) < guardrail_const.PROMPT_INJECTION_MIN_SCAN_WORDS:
            return schemas.ScannerResult(
                scanner_name=self.name,
                detected=False,
                categories=[],
                confidence=1.0,
                risk_score=0.0,
                explanation=f"Not checked: the prompt is shorter than {guardrail_const.PROMPT_INJECTION_MIN_SCAN_WORDS} words.",
            )

        classifier = self.ensure_model_loaded()
        # Chunked rather than truncated: the pipeline caps at 512 tokens, so
        # scoring the prompt in one shot let an attacker bury the payload
        # behind filler and have it dropped before it was ever classified.
        classification = text_chunking.highest_scoring_classification(classifier, prompt, injection_score)
        predicted_label = classification["label"]
        confidence = classification["score"]

        block_threshold = scanner_thresholds.resolve_threshold(scanner_policy, guardrail_const.PROMPT_INJECTION_BLOCK_THRESHOLD, self.name)
        is_injection = predicted_label == guardrail_const.PROMPT_INJECTION_LABEL and confidence >= block_threshold
        if not is_injection:
            return schemas.ScannerResult(
                scanner_name=self.name,
                detected=False,
                categories=[],
                confidence=confidence,
                risk_score=0.0,
                explanation="No prompt injection detected.",
            )

        return schemas.ScannerResult(
            scanner_name=self.name,
            detected=True,
            categories=[guardrail_const.PROMPT_INJECTION_CATEGORY],
            confidence=confidence,
            risk_score=confidence,
            explanation=f"Prompt injection detected: the prompt tries to override the AI's instructions (confidence {confidence:.0%}).",
        )
