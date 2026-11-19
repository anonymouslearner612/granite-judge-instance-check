# Toxicity Scanner -- classifies the prompt with Detoxify's 'original' model,
# a multi-label toxic-comment classifier (toxicity, severe_toxicity, obscene,
# threat, insult, identity_attack), run locally via transformers/torch.
# Loaded as an INT8 dynamic-quantized model (2-3x faster on CPU, negligible
# accuracy loss) -- quantizing happens once at load time, not per scan().
# On ARM64 the qnnpack engine is selected first, and if quantizing fails the
# full-precision model is used instead -- see load_model().
#
# Each label is checked against its own threshold (constants.py) since a
# label like severe_toxicity is meaningful at a lower score than an everyday
# label like toxicity.
#
# This scanner judges whether text is abusive, and nothing else. It used to
# also carry a TOXICITY_KEYWORDS regex list that forced a block on a single
# bare word ("bomb", "kill", "attack", "hack", "weapon") regardless of the
# model's own score, which blocked ordinary technical English while the
# model rated it at ~0.002. Detecting requests for harmful *information* --
# which is what that list was really reaching for, and which Detoxify
# genuinely cannot do -- now lives in the harmful-content RuleScanner.

import os
import platform

import torch
from detoxify import Detoxify

from app.core import settings
from app.guardrail import constants as guardrail_const
from app.guardrail import reason_text
from app.guardrail import scanner_thresholds
from app.guardrail import schemas
from app.guardrail import text_chunking
from app.loggers.applicationlogger import app_logger


def is_arm64():
    return platform.machine().lower() in guardrail_const.ARM64_MACHINE_NAMES


class ToxicityScanner:
    name = guardrail_const.TOXICITY_SCANNER

    def __init__(self):
        self.model = None

    def ensure_model_loaded(self):
        if self.model is None:
            self.model = self.load_model()
        return self.model

    def load_model(self):
        model = self.load_detoxify()

        if is_arm64():
            if guardrail_const.ARM_QUANTIZED_ENGINE not in torch.backends.quantized.supported_engines:
                app_logger.error(
                    f"{guardrail_const.ARM_QUANTIZED_ENGINE} engine is not available on this ARM64 build; "
                    "toxicity model is running unquantized (FP32)",
                    exc_info=False,
                )
                return model
            torch.backends.quantized.engine = guardrail_const.ARM_QUANTIZED_ENGINE

        try:
            model.model = torch.quantization.quantize_dynamic(model.model, {torch.nn.Linear}, dtype=torch.qint8)
        except Exception:
            # Quantizing is a speed optimisation, and this runs in the startup
            # lifespan, where an exception stopped the whole gateway booting on
            # ARM64. quantize_dynamic works on a copy, so model.model is still
            # the intact FP32 model. Logged as an error because FP32 is slower
            # and scores slightly differently from the quantized model the
            # thresholds were tuned on.
            app_logger.exception(
                "Toxicity model could not be quantized; running unquantized (FP32) "
                f"(quantized engine: {torch.backends.quantized.engine})"
            )
            return model

        app_logger.info(f"Toxicity model quantized with engine {torch.backends.quantized.engine}")
        return model

    def load_detoxify(self):
        # Without a checkpoint, Detoxify downloads ~418 MB from GitHub on
        # every fresh container. The mounted checkpoint is the 'original'
        # variant's file, so it must change together with TOXICITY_MODEL_VARIANT.
        checkpoint_path = settings.DETOXIFY_CHECKPOINT_PATH
        if not os.path.exists(checkpoint_path):
            app_logger.warning(f"Detoxify checkpoint not found at {checkpoint_path}; downloading it instead")
            return Detoxify(guardrail_const.TOXICITY_MODEL_VARIANT)

        app_logger.info(f"Detoxify checkpoint loaded from {checkpoint_path}")
        return Detoxify(guardrail_const.TOXICITY_MODEL_VARIANT, checkpoint=checkpoint_path)

    def scan(self, prompt, scanner_policy=None):
        # The policy's only say here is its threshold: Detoxify scores the
        # text as a whole, so there is no span to redact, and the action
        # that decides what a detection means is applied by the engine,
        # not here.
        classifier = self.ensure_model_loaded()
        label_scores = self.highest_label_scores(classifier, prompt)
        threshold_by_label, fallback_threshold, max_score_threshold = self.resolve_thresholds(scanner_policy)

        max_score = max(label_scores.values(), default=0.0)
        highest_label = max(label_scores, key=label_scores.get)
        highest_score = label_scores[highest_label]
        triggered_labels = []
        for label, score in label_scores.items():
            if score >= threshold_by_label.get(label, fallback_threshold):
                triggered_labels.append(label)
        triggered_labels.sort()
        detected = bool(triggered_labels) or max_score >= max_score_threshold

        if not detected:
            return schemas.ScannerResult(
                scanner_name=self.name,
                detected=False,
                categories=[],
                confidence=highest_score,
                risk_score=max_score,
                explanation="No toxic content detected.",
            )

        return schemas.ScannerResult(
            scanner_name=self.name,
            detected=True,
            categories=triggered_labels or [highest_label],
            confidence=highest_score,
            risk_score=max_score,
            explanation=f"Toxic content detected: {reason_text.readable_list(triggered_labels or [highest_label])} ({reason_text.percent(highest_score)}).",
        )

    def resolve_thresholds(self, scanner_policy):
        """
        Returns (threshold_by_label, fallback_threshold, max_score_threshold).

        A policy carries one threshold and this model produces six scores,
        so an operator's single dial replaces every default rather than
        scaling them: "flag anything scoring at least this" is the only
        reading of one number that isn't a guess about how the labels
        relate to each other. Unset, each label keeps its own default --
        severe_toxicity is meaningful at a lower score than toxicity.
        """
        configured_threshold = scanner_thresholds.resolve_threshold(scanner_policy, None, self.name)
        if configured_threshold is None:
            return guardrail_const.TOXICITY_LABEL_THRESHOLDS, guardrail_const.TOXICITY_THRESHOLD_BLOCK, guardrail_const.MAX_SCORE_BLOCK_THRESHOLD

        uniform_thresholds = {}
        for label in guardrail_const.TOXICITY_LABEL_THRESHOLDS:
            uniform_thresholds[label] = configured_threshold
        return uniform_thresholds, configured_threshold, configured_threshold

    def highest_label_scores(self, classifier, prompt):
        # Detoxify's model also caps at 512 tokens, so a long prompt is
        # scored in overlapping windows and each label keeps its highest
        # score across them -- one abusive passage late in a long message
        # still counts.
        highest_scores = {}
        for chunk in text_chunking.split_into_chunks(text_chunking.cap_text_length(prompt)):
            for label, score in classifier.predict(chunk).items():
                score = float(score)
                if label not in highest_scores or score > highest_scores[label]:
                    highest_scores[label] = score
        return highest_scores
