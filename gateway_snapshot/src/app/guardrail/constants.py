import os

from app.utils.runtime_parsers import runtime_constants as runtime_const


APPLICATION_TYPE_AGENT = "agent"

SECRET_SCANNER = "secret_scanner"
PII_SCANNER = "pii_scanner"
PROMPT_INJECTION_SCANNER = "prompt_injection_scanner"
TOXICITY_SCANNER = "toxicity_scanner"
KNOWN_ATTACK_PATTERN_SCANNER = "known_attack_pattern_scanner"
DANGEROUS_PATTERN_SCANNER = "dangerous_pattern_scanner"
LANGUAGE_SCANNER = "language_scanner"
MALWARE_SCANNER = "malware_scanner"
HARMFUL_CONTENT_SCANNER = "harmful_content_scanner"
MALICIOUS_ENTITY_SCANNER = "malicious_entity_scanner"

CUSTOM_DETECTOR_BLOCK_SCANNER = "custom_detector_block_scanner"
CUSTOM_DETECTOR_REDACT_SCANNER = "custom_detector_redact_scanner"
CUSTOM_DETECTOR_MONITOR_SCANNER = "custom_detector_monitor_scanner"
CUSTOM_DETECTOR_SCANNERS = [CUSTOM_DETECTOR_BLOCK_SCANNER, CUSTOM_DETECTOR_REDACT_SCANNER, CUSTOM_DETECTOR_MONITOR_SCANNER]

SCANNER_DISPLAY_NAMES = {
    SECRET_SCANNER: "Secret",
    PII_SCANNER: "PII",
    PROMPT_INJECTION_SCANNER: "Prompt Injection",
    TOXICITY_SCANNER: "Toxicity",
    KNOWN_ATTACK_PATTERN_SCANNER: "Known Attack Pattern",
    DANGEROUS_PATTERN_SCANNER: "Dangerous Pattern",
    LANGUAGE_SCANNER: "Language",
    MALWARE_SCANNER: "Malware",
    HARMFUL_CONTENT_SCANNER: "Harmful Content",
    CUSTOM_DETECTOR_BLOCK_SCANNER: "Custom Detector",
    CUSTOM_DETECTOR_REDACT_SCANNER: "Custom Detector",
    CUSTOM_DETECTOR_MONITOR_SCANNER: "Custom Detector",
}

# --- Reason text (reason_text.py) ---
#
# Each check's explanation becomes the Reason users read, so internal ids
# (entity types, model labels, rule and YARA names) are shown through this
# table. An id not listed falls back to its words ("SOME_TYPE" -> "Some type").
REASON_LABELS = {
    # Credentials (Secret Scanner, and the PII Scanner's credential patterns)
    "OPENAI_API_KEY": "OpenAI API key",
    "ANTHROPIC_API_KEY": "Anthropic API key",
    "GITHUB_TOKEN": "GitHub token",
    "GITHUB_PAT": "GitHub personal access token",
    "SLACK_TOKEN": "Slack token",
    "STRIPE_SECRET_KEY": "Stripe secret key",
    "GOOGLE_API_KEY": "Google API key",
    "API_TOKEN": "AWS access key ID",
    "BEARER_TOKEN": "Bearer token",
    "GENERIC_TOKEN": "Access token",
    "AWS_SECRET_KEY": "AWS secret access key",
    "JWT_TOKEN": "JSON Web Token",
    "PRIVATE_KEY_BLOCK": "Private key",
    "SESSION_ID": "Session ID",
    "PASSWORD": "Password",
    "PRODUCT_KEY": "Product key",
    # Personal information (PII Scanner)
    "PERSON": "Person name",
    "EMAIL_ADDRESS": "Email address",
    "PHONE_NUMBER": "Phone number",
    "CUSTOM_PHONE": "Phone number",
    "PHONE_INTL": "Phone number",
    "CREDIT_CARD": "Credit card number",
    "CRYPTO": "Crypto wallet address",
    "IBAN_CODE": "IBAN",
    "IBAN": "IBAN",
    "IP_ADDRESS": "IP address",
    "MAC_ADDRESS": "MAC address",
    "LOCATION": "Location",
    "DATE_TIME": "Date or time",
    "NRP": "Nationality, religion or political group",
    "URL": "URL",
    "MEDICAL_LICENSE": "Medical license number",
    "UK_NHS": "UK NHS number",
    "US_SSN": "US Social Security number",
    "US_ITIN": "US ITIN",
    "US_PASSPORT": "US passport number",
    "US_DRIVER_LICENSE": "US driver's license number",
    "US_BANK_NUMBER": "US bank account number",
    "IN_AADHAAR": "Aadhaar number",
    "IN_PAN": "PAN number",
    # Toxicity Scanner labels
    "toxicity": "Toxic language",
    "severe_toxicity": "Severe toxicity",
    "obscene": "Obscene language",
    "threat": "Threat",
    "insult": "Insult",
    "identity_attack": "Identity-based attack",
    "sexual_explicit": "Sexually explicit content",
    # Malware Scanner (YARA rule names in rules/malware_patterns.yar)
    "webshell_php_eval": "PHP web shell",
    "reverse_shell_command": "Reverse shell command",
    "obfuscated_powershell": "Obfuscated PowerShell",
    "sql_injection_payload": "SQL injection payload",
    "credential_dumping_tool": "Credential dumping tool",
    # Rule scanners (HARMFUL_REQUEST_PATTERNS, KNOWN_ATTACK_PATTERNS, DANGEROUS_COMMAND_PATTERNS)
    "weapons_explosives": "Weapons or explosives",
    "illicit_drugs": "Illicit drugs",
    "chemical_biological": "Chemical or biological weapons",
    "targeted_violence": "Targeted violence",
    "social_engineering": "Social engineering",
    "self_harm": "Self-harm",
    "human_exploitation": "Human exploitation",
    "dan_mode": "\"DAN mode\" jailbreak",
    "do_anything_now": "\"Do anything now\" jailbreak",
    "ignore_previous_instructions": "\"Ignore previous instructions\"",
    "disregard_previous_instructions": "\"Disregard previous instructions\"",
    "developer_mode": "\"Developer mode\" jailbreak",
    "shell_destructive": "Destructive shell command (rm -rf /)",
    "shell_fork_bomb": "Shell fork bomb",
    "shell_reverse_shell": "Reverse shell command",
    "shell_curl_pipe_bash": "Download piped into a shell",
    "shell_privilege_escalation": "Unsafe permission change (chmod 777 /)",
    "sql_drop_table": "SQL DROP TABLE",
    "sql_truncate_table": "SQL TRUNCATE TABLE",
}

# Codes the language model reports (papluca/xlm-roberta-base-language-detection).
LANGUAGE_NAMES = {
    "ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "fr": "French", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
    "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sw": "Swahili",
    "th": "Thai", "tr": "Turkish", "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese",
}

HIDDEN_CONTENT_REASON_SUFFIX = " (found in hidden or encoded text)."

# Hard ceiling on the text any scanner is handed. Nothing capped prompt
# length before, so a multi-megabyte prompt could hold Presidio's NER open
# for an unbounded time. Text beyond this is dropped before scanning, which
# is safe only because the ML scanners now chunk (see
# PROMPT_INJECTION_MAX_CHUNKS) rather than silently truncating at 512
# tokens the way they used to.
MAX_SCAN_TEXT_CHARS = 200000

# --- Scanner execution budget (guardrail_engine.py) ---
#
# Wall-clock ceiling per scanner call. Nothing bounded this before, so a
# pathological input to Presidio's NER or a wedged model could hold a
# request open indefinitely. On expiry the scanner is treated exactly like
# one that raised: recorded as failed, contributing ALLOW, logged.
#
# 15s is far above any healthy scan (the slowest measured is ~4s, a heavily
# chunked long prompt) -- this is a backstop against a hang, not a
# performance target.
#
# A timed-out scanner's thread is not killed; Python cannot safely kill one.
# It keeps running until it finishes and its result is discarded, so the
# pool below is sized to absorb a few of those without starving live
# requests.
#
# The clock covers lazy model loading, since that happens inside the first
# scan() call. That is safe only because boot.py preloads every model at
# startup -- without that, the first request to reach a cold ML scanner
# could spend the whole budget loading weights and be recorded as a
# timeout. Keep preload_models() wired into startup.
SCANNER_TIMEOUT_SECONDS = 15

# How long a scanner may sit in the shared pool waiting for a free worker
# before it is given up on. Queueing and executing get separate budgets on
# purpose: the clock used to start at submission, so a scanner that waited
# out the whole budget without ever running was recorded as a timeout --
# and a timeout resolves to ALLOW for every modifying and logging scanner,
# so load alone could quietly turn the guardrail off. Reaching this means
# the pool is saturated (see SCANNER_THREAD_POOL_SIZE), not that a scanner
# is slow, and the two are now reported differently.
SCANNER_QUEUE_WAIT_SECONDS = 10

# One shared pool for all scanner execution instead of a new
# ThreadPoolExecutor per request. Per-request pools meant N concurrent
# requests spawned N x scanner-count threads with no ceiling.
#
# Size this to the container's CPU quota: one scan submits seven to nine
# jobs, so a pool of 32 saturates at roughly four concurrent scans. Past
# saturation, scanners wait in the queue and are given up on
# (SCANNER_QUEUE_WAIT_SECONDS), which for a modifying or logging scanner
# means allowing what it was meant to catch.
SCANNER_THREAD_POOL_SIZE = 32

# How many threads torch may use for one model inference.
#
# Torch defaults to one intra-op thread per visible core, and several
# scanners run inference at once in the pool above -- so on a container
# with a small CPU quota, each concurrent inference tries to use every
# core and they thrash. A low cap makes concurrent scans slower
# individually and much faster in aggregate, which is the trade a shared
# pool is for.
TORCH_THREADS = 2

ARM64_MACHINE_NAMES = ["aarch64", "arm64"]
ARM_QUANTIZED_ENGINE = "qnnpack"

DETERMINISTIC_SCANNERS = [
    SECRET_SCANNER,
    PII_SCANNER,
    MALWARE_SCANNER,
    MALICIOUS_ENTITY_SCANNER,
] + CUSTOM_DETECTOR_SCANNERS

# --- Hidden-content pass (guardrail_engine.run_hidden_content_pass) ---
#
# Decoded Base64/hex/URL fragments are scanned separately from the prompt
# they were pulled out of, and any 20-character base64-looking run triggers
# that pass. Hashes, blobs and data-URIs are ordinary in developer traffic,
# so without a limit an attacker doubles the cost of every scan -- models
# included -- by pasting one. Two limits keep the pass cheap: only these
# scanners run over fragments, and only this much decoded text is scanned.
#
# Detoxify and the language model are left out because a rude decoded
# fragment is not the threat model. What matters inside an encoding is a
# credential, an injected instruction, or a known attack pattern -- and the
# pass keeps blocks only, so a finding that cannot block is wasted work.
HIDDEN_CONTENT_SCANNERS = DETERMINISTIC_SCANNERS + [PROMPT_INJECTION_SCANNER]
MAX_HIDDEN_CONTENT_CHARS = 10000

DIRECTION_INPUT_SCAN = "input"

# --- Secret Scanner (secret_scanner.py, detect-secrets) ---
#
# Excluded because these plugins flag ordinary chat text as "secrets":
# entropy-based plugins fire on any short mixed-case/alphanumeric word, and
# IPPublicDetector fires on any public IPv4 mention.
SECRET_SCANNER_EXCLUDED_PLUGINS = ["Base64HighEntropyString", "HexHighEntropyString", "IPPublicDetector"]

# --- PII Scanner (pii_scanner.py, Microsoft Presidio) ---

# Entity types PIIScanner detects. Comment one out to disable it -- DATE_TIME
# and NRP start disabled since they fire on ordinary words ("tomorrow",
# "American", "Christian") that aren't PII in any practical sense.
PII_SCANNER_INCLUDED_ENTITIES = [
    "CREDIT_CARD",
    "CRYPTO",
    # "DATE_TIME",  # fires on ordinary words like "tomorrow"
    "EMAIL_ADDRESS",
    "IBAN_CODE",
    "IP_ADDRESS",
    # "LOCATION",  # fires on any place name -- "Kyoto", "Roman Empire"
    "MAC_ADDRESS",
    "MEDICAL_LICENSE",
    # "NRP",  # fires on ordinary words like "American", "Christian"
    # "PERSON",
    "PHONE_NUMBER",
    "UK_NHS",
    "URL",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_ITIN",
    "US_PASSPORT",
    "US_SSN",
]

# Sub-rule entities for Personal Data (PII)
PII_ENTITY_EMAIL = "email"
PII_ENTITY_CREDIT_CARD = "credit_card"
PII_ENTITY_PHONE_NUMBER = "phone_number"
PII_ENTITY_SSN = "ssn"
PII_ENTITY_BANK_ACCOUNT = "bank_account"
PII_ENTITY_GOVERNMENT_ID = "government_id"

# Sub-rule entities for Secrets and Credentials
SECRET_ENTITY_API_KEY = "api_key"
SECRET_ENTITY_CLOUD_CREDENTIAL = "cloud_credential"
SECRET_ENTITY_ACCESS_TOKEN = "access_token"
SECRET_ENTITY_PRIVATE_KEY = "private_key"
SECRET_ENTITY_PASSWORD = "password"
SECRET_ENTITY_AUTH_TOKEN = "auth_token"

# Sub-rule entities for Malicious Entity
MALICIOUS_ENTITY_IP_ADDRESS = "ip_address"
MALICIOUS_ENTITY_URL = "url"
MALICIOUS_ENTITY_DOMAIN = "domain"
MALICIOUS_ENTITY_FILE_HASH = "file_hash"
MALICIOUS_ENTITY_CVE_ID = "cve_id"

PII_SUB_RULE_ENTITY_MAPPING = {
    PII_ENTITY_EMAIL: ["EMAIL_ADDRESS"],
    PII_ENTITY_CREDIT_CARD: ["CREDIT_CARD"],
    PII_ENTITY_PHONE_NUMBER: ["PHONE_NUMBER", "CUSTOM_PHONE", "PHONE_INTL"],
    PII_ENTITY_SSN: ["US_SSN", "US_ITIN"],
    PII_ENTITY_BANK_ACCOUNT: ["US_BANK_NUMBER", "IBAN_CODE", "IBAN", "CRYPTO"],
    PII_ENTITY_GOVERNMENT_ID: ["US_PASSPORT", "US_DRIVER_LICENSE", "IN_AADHAAR", "IN_PAN"],
}

SECRET_SUB_RULE_ENTITY_MAPPING = {
    SECRET_ENTITY_API_KEY: [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "STRIPE_SECRET_KEY",
        "GENERIC_TOKEN",
        "OpenAIDetector",
        "StripeDetector",
        "MailchimpDetector",
        "SquareOAuthDetector",
        "TwilioKeyDetector",
        "SendGridDetector",
    ],
    SECRET_ENTITY_CLOUD_CREDENTIAL: [
        "AWS_SECRET_KEY",
        "API_TOKEN",
        "AWSKeyDetector",
        "AzureStorageKeyDetector",
        "IbmCloudIamDetector",
        "IbmCosHmacDetector",
        "SoftlayerDetector",
    ],
    SECRET_ENTITY_ACCESS_TOKEN: [
        "GITHUB_TOKEN",
        "GITHUB_PAT",
        "SLACK_TOKEN",
        "BEARER_TOKEN",
        "GitHubTokenDetector",
        "SlackDetector",
        "NpmDetector",
        "PypiTokenDetector",
        "DiscordBotTokenDetector",
        "TelegramBotTokenDetector",
        "ArtifactoryDetector",
    ],
    SECRET_ENTITY_PRIVATE_KEY: [
        "PRIVATE_KEY_BLOCK",
        "PrivateKeyDetector",
    ],
    SECRET_ENTITY_PASSWORD: [
        "PASSWORD",
        "BasicAuthDetector",
        "KeywordDetector",
    ],
    SECRET_ENTITY_AUTH_TOKEN: [
        "JWT_TOKEN",
        "SESSION_ID",
        "JwtTokenDetector",
        "CloudantDetector",
    ],
}

# Regex patterns for PII/secret shapes Presidio's NER doesn't cover, run as a
# second detection layer alongside it. The token-shaped entries listed in
# SECRET_SCANNER_CUSTOM_PATTERN_TYPES are read only by SecretScanner; PIIScanner
# skips them so each credential follows the Secret Scanner's action alone.
# Patterns deliberately NOT here, because each fired on ordinary business
# text often enough to block legitimate traffic (all measured, see the
# audit): BANK_ACCOUNT (\d{8,18} matched any order number, timestamp or
# amount),
# SWIFT_CODE (matched any 8-character uppercase word, e.g. "SECURITY",
# "RESOLVED"), PRODUCT_KEY (any three hyphenated 4-character groups),
# INTERNAL_ID, and the three DOB_* patterns (every ISO or numeric date in
# every prompt). A real bank-account or SWIFT detector needs surrounding
# context, not a bare digit/letter run -- add it back only with that context.
CUSTOM_PATTERNS = {
    # OCR can insert horizontal whitespace around a domain dot. Keep the
    # original span so both prompt redaction and audit masking cover that whitespace.
    "EMAIL_ADDRESS": (
        r"(?i)(?<![a-z0-9._%+@-])"
        r"[a-z0-9_%+-]+(?:\.[a-z0-9_%+-]+)*@"
        r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
        r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
        r"(?:[ \t]+\.[ \t]*|\.[ \t]+)"
        r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
        r"[a-z]{2,63}(?![a-z0-9_-])"
    ),
    "PRODUCT_KEY": r"\b[A-Z0-9]{4}(?:-[A-Z0-9]{4}){2}\b",
    "OPENAI_API_KEY": r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{9,}\b",
    "ANTHROPIC_API_KEY": r"\bsk-ant-[A-Za-z0-9\-_]{16,}\b",
    "GITHUB_TOKEN": r"\bgh[pousr]_[A-Za-z0-9]{20,255}\b",
    "GITHUB_PAT": r"\bgithub_pat_[A-Za-z0-9_]{20,255}\b",
    "SLACK_TOKEN": r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",
    "STRIPE_SECRET_KEY": r"\bsk_(?:live|test)_[A-Za-z0-9]{16,}\b",
    "GOOGLE_API_KEY": r"\bAIza[0-9A-Za-z\-_]{35}\b",
    "API_TOKEN": r"\b(?:AKIA|ASIA)[0-9A-Z]{8,20}\b",
    "BEARER_TOKEN": r"(?i)\bbearer\s+([A-Za-z0-9\-._=]{20,})",
    "GENERIC_TOKEN": r"(?i)\b(?:access|auth|refresh|id)[_-]?token\b\s*(?:is|=|:)\s*['\"]?([A-Za-z0-9\-._=]{12,})",
    "AWS_SECRET_KEY": r"(?i)\baws[_ -]?secret[_ -]?access[_ -]?key\b\s*(?:is|=|:)\s*['\"]?([A-Za-z0-9/+=]{20,60})",
    "JWT_TOKEN": r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b",
    "PRIVATE_KEY_BLOCK": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    "SESSION_ID": r"(?i)\bsess-[a-z0-9]{16,64}\b",
    "CUSTOM_PHONE": r"\b\d{3,4}[-.]\d{3}[-.]\d{3,4}\b",
    "PHONE_INTL": r"\+\d{1,3}[\s.-]?\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{3,4}\b",
    "US_SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "IN_AADHAAR": r"(?<!\d)(?<!\d\s)\b\d{4}\s\d{4}\s\d{4}\b(?!\s?\d)",
    "IN_PAN": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "IBAN": r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
    "PASSWORD": r"(?i)\b(?:password|passcode|pwd)\s*(?:is|=|:)\s*['\"]?([^\s'\";,]{6,})",
}

# Per-pattern detection confidence for CUSTOM_PATTERNS, read by
# pii_scanner.detect_custom_entities. Previously every custom pattern was
# scored at a flat 0.95, which cleared every tier in min_score_for_entity
# and made the PRESIDIO_ENTITY_WEIGHTS system dead code for regex hits. A
# pattern with a distinctive vendor prefix (sk-ant-, ghp_, AIza) is close to
# certain; a bare structural shape is not. A type absent here falls back to
# DEFAULT_CUSTOM_PATTERN_SCORE.
CUSTOM_PATTERN_SCORES = {
    "EMAIL_ADDRESS": 0.85,
    "ANTHROPIC_API_KEY": 0.99,
    "GITHUB_TOKEN": 0.99,
    "GITHUB_PAT": 0.99,
    "SLACK_TOKEN": 0.99,
    "STRIPE_SECRET_KEY": 0.99,
    "GOOGLE_API_KEY": 0.99,
    "PRIVATE_KEY_BLOCK": 0.99,
    "JWT_TOKEN": 0.95,
    "OPENAI_API_KEY": 0.95,
    "API_TOKEN": 0.95,
    "AWS_SECRET_KEY": 0.95,
    "GENERIC_TOKEN": 0.9,
    "BEARER_TOKEN": 0.9,
    "SESSION_ID": 0.9,
    "US_SSN": 0.95,
    "IN_PAN": 0.9,
    "IBAN": 0.85,
    "IN_AADHAAR": 0.8,
    "PASSWORD": 0.8,
    # Structural shapes with no distinguishing prefix -- a phone-shaped run
    # of digits is frequently something else.
    "CUSTOM_PHONE": 0.6,
    "PHONE_INTL": 0.6,
}
DEFAULT_CUSTOM_PATTERN_SCORE = 0.7

# The subset of CUSTOM_PATTERNS keys that are vendor/token secrets, as
# opposed to the PII-shaped entries in the same dict (US_SSN, IN_AADHAAR,
# IN_PAN, CUSTOM_PHONE, ...). Read by SecretScanner as a second detection
# layer alongside detect-secrets' own plugins -- detect-secrets ships no
# plugin for several of these vendors (e.g. Anthropic keys), and its only
# generic fallback for an arbitrary opaque token (entropy-based plugins) is
# excluded (see SECRET_SCANNER_EXCLUDED_PLUGINS) for flagging ordinary words.
# This closes that gap using patterns PIIScanner already carries, rather
# than duplicating the regexes in a second dict. PIIScanner skips every type
# listed here, so a credential is reported by this scanner only.
SECRET_SCANNER_CUSTOM_PATTERN_TYPES = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GITHUB_TOKEN",
    "GITHUB_PAT",
    "SLACK_TOKEN",
    "STRIPE_SECRET_KEY",
    "GOOGLE_API_KEY",
    "API_TOKEN",
    "BEARER_TOKEN",
    "GENERIC_TOKEN",
    "AWS_SECRET_KEY",
    "JWT_TOKEN",
    "PRIVATE_KEY_BLOCK",
    "SESSION_ID",
    "PASSWORD",
]

# Entity types named as credentials, not personal information, in the PII
# Scanner's reason. PRODUCT_KEY is the only one it still reports itself.
CREDENTIAL_ENTITY_TYPES = SECRET_SCANNER_CUSTOM_PATTERN_TYPES + ["PRODUCT_KEY"]

# Per-entity-type weight, used by pii_scanner.min_score_for_entity to pick a
# stricter or looser confidence cutoff per entity type (a missed low-weight
# entity like a URL is cheap; a missed high-weight one like a credit card
# number is not). An entity type not listed here falls back to weight 0.5 --
# the strictest tier -- since an unrecognized type is unproven.
PRESIDIO_ENTITY_WEIGHTS = {
    # --- Critical / financial / secrets ---
    "PASSWORD": 1.0,
    "CREDIT_CARD": 1.0,
    "IBAN_CODE": 1.0,
    "US_BANK_NUMBER": 1.0,
    "US_SSN": 1.0,
    "US_ITIN": 1.0,
    "US_PASSPORT": 1.0,
    "US_DRIVER_LICENSE": 1.0,
    "CRYPTO": 1.0,
    "OPENAI_API_KEY": 1.0,
    "ANTHROPIC_API_KEY": 1.0,
    "GITHUB_TOKEN": 1.0,
    "GITHUB_PAT": 1.0,
    "SLACK_TOKEN": 1.0,
    "STRIPE_SECRET_KEY": 1.0,
    "GOOGLE_API_KEY": 1.0,
    "BEARER_TOKEN": 1.0,
    "GENERIC_TOKEN": 1.0,
    "JWT_TOKEN": 1.0,
    "PRIVATE_KEY_BLOCK": 1.0,
    "IN_AADHAAR": 1.0,
    "IN_PAN": 1.0,

    # --- High / personally identifiable ---
    "EMAIL_ADDRESS": 0.8,
    "PHONE_NUMBER": 0.8,

    # --- Medium / identifiers, network, misc ---
    "IP_ADDRESS": 0.6,
    "MAC_ADDRESS": 0.6,
    "URL": 0.6,
    "MEDICAL_LICENSE": 0.6,
}
DEFAULT_PRESIDIO_ENTITY_WEIGHT = 0.5

# Minimum model confidence a PII detection needs, by its entity weight above:
# (highest weight in the tier, minimum score). Weightier types are trusted at a
# lower confidence, since missing one costs more than a false positive.
PII_MIN_SCORE_TIERS = [
    (0.4, 0.9),
    (0.6, 0.8),
]
PII_MIN_SCORE_FALLBACK = 0.7

PII_ANALYZER_LANGUAGE = "en"
PII_NLP_ENGINE_NAME = "spacy"

# Stripped before Presidio sees the text, so an invisible character can't split an entity.
ZERO_WIDTH_CHARS_PATTERN = r"[​-‏‪-‮﻿]"
CONTROL_CHARS_PATTERN = r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"

# Token limit of the transformer models the classifier scanners load.
CLASSIFIER_MAX_TOKENS = 512

# --- Prompt Injection Scanner (prompt_injection_scanner.py) ---
PROMPT_INJECTION_MODEL_NAME = "patronus-studio/wolf-defender-prompt-injection"
UNTUNED_PROMPT_INJECTION_THRESHOLD = 0.5

PROMPT_INJECTION_MODEL_SETTINGS = {
    # --- Tuned: thresholds measured against real traffic ---
    "proventra/mdeberta-v3-base-prompt-injection": {"label": "INJECTION", "threshold": 0.75},
    "protectai/deberta-v3-base-prompt-injection-v2": {"label": "INJECTION", "threshold": 0.95},
    "bastionsoft/binary-bastion-prompt-protection-deberta-v3-xsmall-v1": {"label": "attack", "threshold": 0.55},
    "marklkelly/bert-tiny-injection-detector": {"label": "INJECTION", "threshold": 0.65},

    # --- Untuned candidates, benchmarked but never run in the gateway ---
    "meta-llama/Llama-Prompt-Guard-2-86M": {"label": "LABEL_1", "threshold": UNTUNED_PROMPT_INJECTION_THRESHOLD},
    "meta-llama/Llama-Prompt-Guard-2-22M": {"label": "LABEL_1", "threshold": UNTUNED_PROMPT_INJECTION_THRESHOLD},
    "meta-llama/Prompt-Guard-86M": {"label": "LABEL_1", "threshold": UNTUNED_PROMPT_INJECTION_THRESHOLD},
    "bastionsoft/binary-bastion-prompt-protection-mdeberta-v3-base-v1": {"label": "attack", "threshold": 0.55},
    "patronus-studio/wolf-defender-prompt-injection": {"label": "injection", "threshold": UNTUNED_PROMPT_INJECTION_THRESHOLD},
    "patronus-studio/wolf-defender-prompt-injection-small": {"label": "injection", "threshold": UNTUNED_PROMPT_INJECTION_THRESHOLD},
    "protectai/deberta-v3-base-prompt-injection": {"label": "INJECTION", "threshold": UNTUNED_PROMPT_INJECTION_THRESHOLD},
    "protectai/deberta-v3-small-prompt-injection-v2": {"label": "INJECTION", "threshold": UNTUNED_PROMPT_INJECTION_THRESHOLD},
    "madhurjindal/Jailbreak-Detector": {"label": "jailbreak", "threshold": UNTUNED_PROMPT_INJECTION_THRESHOLD},
    "codeintegrity-ai/promptguard": {"label": "INJECTION", "threshold": UNTUNED_PROMPT_INJECTION_THRESHOLD},
    "llm-semantic-router/mmbert32k-jailbreak-detector-merged": {"label": "jailbreak", "threshold": UNTUNED_PROMPT_INJECTION_THRESHOLD},
    # Ships no id2label at all, so LABEL_1 is a guess -- verify it against the
    # model card before reading anything into a score from this one.
    "fmops/distilbert-prompt-injection": {"label": "LABEL_1", "threshold": UNTUNED_PROMPT_INJECTION_THRESHOLD},
    "testsavantai/prompt-injection-defender-large-v0": {"label": "INJECTION", "threshold": UNTUNED_PROMPT_INJECTION_THRESHOLD},
    "qualifire/prompt-injection-sentinel": {"label": "jailbreak", "threshold": UNTUNED_PROMPT_INJECTION_THRESHOLD},
    "rogue-security/prompt-injection-jailbreak-sentinel-v2": {"label": "jailbreak", "threshold": UNTUNED_PROMPT_INJECTION_THRESHOLD},
    "jackhhao/jailbreak-classifier": {"label": "jailbreak", "threshold": UNTUNED_PROMPT_INJECTION_THRESHOLD},
    "deepset/deberta-v3-base-injection": {"label": "INJECTION", "threshold": UNTUNED_PROMPT_INJECTION_THRESHOLD},
    # XLM-RoBERTa, so genuinely multilingual -- the property proventra was
    # picked for, and the one to try if non-English prompts are the concern.
    "RyanStudio/Mezzo-Prompt-Guard-v2-Base": {"label": "unsafe", "threshold": UNTUNED_PROMPT_INJECTION_THRESHOLD},
    "devndeploy/bert-prompt-injection-detector": {"label": "INJECTION", "threshold": UNTUNED_PROMPT_INJECTION_THRESHOLD},
    "fredstrey/deberta-v3-base-prompt-injection-v3": {"label": "INJECTION", "threshold": UNTUNED_PROMPT_INJECTION_THRESHOLD},
}

PROMPT_INJECTION_LABEL = PROMPT_INJECTION_MODEL_SETTINGS[PROMPT_INJECTION_MODEL_NAME]["label"]
PROMPT_INJECTION_BLOCK_THRESHOLD = PROMPT_INJECTION_MODEL_SETTINGS[PROMPT_INJECTION_MODEL_NAME]["threshold"]

# Below this word count there isn't enough context for the classifier to mean
# anything (it's tuned on sentence-length text) -- skip the model call
# entirely and allow, rather than spending a GPU/CPU inference on "hi".
PROMPT_INJECTION_MIN_SCAN_WORDS = 3
PROMPT_INJECTION_CATEGORY = "prompt_injection"

# --- Chunking for the transformer-backed scanners (text_chunking.py) ---
#
# These models take 512 tokens. They were built with truncation=True, which
# silently dropped everything past that -- so prefixing ~800 tokens of
# filler hid an injection payload completely (measured: the same attack
# scored INJECTION 1.00 alone and ALLOW 0.61 behind filler). Text is now
# split into overlapping word windows and every window is classified, with
# the highest score winning.
#
# 320 words is comfortably inside 512 tokens for English prose even at a
# pessimistic ~1.5 tokens/word. The overlap exists so a payload straddling
# a window boundary still appears whole in one of them.
SCAN_CHUNK_WORDS = 320
SCAN_CHUNK_OVERLAP_WORDS = 60

# Ceiling on windows classified per scan, so an enormous prompt costs
# bounded time.
#
# 24 windows at a step of 260 words is about 6,300 words of text -- far
# less than the ~30,000 words MAX_SCAN_TEXT_CHARS admits, and the comment
# here used to claim ~20k, which was simply wrong. Those windows are no
# longer taken from the head: they are spread evenly across the whole text
# (text_chunking.select_window_starts), so a payload at word 20,000 is
# reachable where before nothing past ~6,300 was ever classified.
#
# A fixed budget over unbounded text cannot cover everything, so gaps
# between windows remain. Raising this is the real fix and it is affordable
# only once chunk inference is batched into one call per scanner rather
# than one call per window.
MAX_SCAN_CHUNKS = 24

# Sits between the head and the tail of a capped prompt
# (text_chunking.cap_text_length). It exists to stop the join inventing
# text: without it the last words of the head run straight into the first
# words of the tail and can spell a match neither half contains.
SCAN_TRUNCATION_MARKER = " [... truncated ...] "

# --- Toxicity Scanner (toxicity_scanner.py, Detoxify) ---
#
# 'original' is Detoxify's English toxic-comment-classification model,
# scoring six labels: toxicity, severe_toxicity, obscene, threat, insult,
# identity_attack. Each label gets its own block threshold rather than one
# flat cutoff, since a label like severe_toxicity is meaningful at a lower
# score than an everyday label like toxicity.
TOXICITY_MODEL_VARIANT = "original"
TOXICITY_THRESHOLD_BLOCK = 0.6
SEVERE_TOXICITY_THRESHOLD_BLOCK = 0.5
OBSCENE_THRESHOLD_BLOCK = 0.6
INSULT_THRESHOLD_BLOCK = 0.6
IDENTITY_ATTACK_THRESHOLD_BLOCK = 0.6

# If any single label crosses this ceiling it blocks, regardless of the
# per-label thresholds above.
MAX_SCORE_BLOCK_THRESHOLD = 0.75

TOXICITY_LABEL_THRESHOLDS = {
    "toxicity": TOXICITY_THRESHOLD_BLOCK,
    "severe_toxicity": SEVERE_TOXICITY_THRESHOLD_BLOCK,
    "obscene": OBSCENE_THRESHOLD_BLOCK,
    "insult": INSULT_THRESHOLD_BLOCK,
    "identity_attack": IDENTITY_ATTACK_THRESHOLD_BLOCK,
}

# TOXICITY_KEYWORDS used to live here: a list of bare nouns (bomb, kill,
# attack, hack, fraud, weapon, steal, hate, mock, shame) that forced a BLOCK
# on a single match, independent of the model's score. It blocked ordinary
# technical and business English -- "kill a hung process", "cyber attack",
# "fraud detection models" -- while the model itself scored those at ~0.002.
# It was removed, and the harmful-request detection it was really doing
# moved to HARMFUL_REQUEST_PATTERNS below, which matches intent rather than
# vocabulary.

# --- Harmful Content Scanner (rule_scanner.py) ---
#
# Detoxify classifies toxic *language*, not harmful *requests*: it scores
# "how to build a bomb" at 0.014 and "how do I synthesize methamphetamine"
# at 0.001, because neither is rude. That left a real gap, which the old
# TOXICITY_KEYWORDS list was papering over at the cost of a large
# false-positive rate.
#
# These patterns match request *intent* -- an ask for operational detail
# about causing harm -- so they need the verb and the object together, not
# either alone. This is a deterministic stand-in for a trained
# harmful-content classifier, and it only catches plainly-worded requests.
# It is meant to be replaced with a real model.
HARMFUL_CONTENT_CATEGORY = "harmful_content"
HARMFUL_REQUEST_PATTERNS = {
    "weapons_explosives": (
        r"(?i)\b(?:how\s+(?:do|can|would)\s+i|how\s+to|instructions?\s+(?:on|for)|steps?\s+to|guide\s+to|teach\s+me\s+to|show\s+me\s+how\s+to)\b"
        r"[^.?!]{0,60}?\b(?:build|make|construct|assemble|manufacture|synthesi[sz]e|create)\b"
        r"[^.?!]{0,40}?\b(?:bomb|explosive|pipe\s*bomb|ied|grenade|napalm|thermite|molotov|silencer|untraceable\s+(?:gun|firearm)|ghost\s+gun)\b"
    ),
    "illicit_drugs": (
        r"(?i)\b(?:how\s+(?:do|can|would)\s+i|how\s+to|instructions?\s+(?:on|for)|steps?\s+to|guide\s+to|teach\s+me\s+to|recipe\s+for)\b"
        r"[^.?!]{0,60}?\b(?:make|cook|synthesi[sz]e|manufacture|produce|extract)\b"
        r"[^.?!]{0,40}?\b(?:meth|methamphetamine|fentanyl|heroin|cocaine|lsd|mdma|crack\s+cocaine)\b"
    ),
    "chemical_biological": (
        r"(?i)\b(?:how\s+(?:do|can|would)\s+i|how\s+to|instructions?\s+(?:on|for)|steps?\s+to|guide\s+to)\b"
        r"[^.?!]{0,60}?\b(?:make|produce|synthesi[sz]e|weaponi[sz]e|culture|disperse)\b"
        r"[^.?!]{0,40}?\b(?:nerve\s+agent|sarin|vx\s+gas|ricin|anthrax|mustard\s+gas|chlorine\s+gas|bioweapon)\b"
    ),
    "targeted_violence": (
        r"(?i)\b(?:how\s+(?:do|can|would)\s+i|how\s+to|best\s+way\s+to|help\s+me)\b"
        r"[^.?!]{0,40}?\b(?:kill|murder|assassinate|poison|maim|torture|abduct|kidnap)\b"
        r"[^.?!]{0,30}?\b(?:someone|somebody|a\s+person|my\s+\w+|him|her|them|people)\b"
    ),
    "social_engineering": (
        r"(?i)\b(?:write|draft|compose|generate|create|craft)\b"
        r"[^.?!]{0,40}?\b(?:phishing|spear[\s-]?phishing|scam|fraudulent)\b"
        r"[^.?!]{0,30}?\b(?:email|message|text|sms|letter|page|site)\b"
    ),
    "self_harm": (
        r"(?i)\b(?:how\s+(?:do|can)\s+i|how\s+to|best\s+way\s+to|easiest\s+way\s+to)\b"
        r"[^.?!]{0,40}?\b(?:kill\s+myself|commit\s+suicide|end\s+my\s+life|hurt\s+myself|overdose)\b"
    ),
    "human_exploitation": (
        r"(?i)\b(?:how\s+(?:do|can)\s+i|how\s+to|help\s+me|best\s+way\s+to)\b"
        r"[^.?!]{0,50}?\b(?:human\s+traffick|sex\s+traffick|smuggle\s+(?:people|children)|groom\s+a\s+(?:child|minor))\w*"
    ),
}

# --- Known Attack Pattern Scanner (rule_scanner.py) ---
#
# A plain phrase/substring deny-list for publicly documented jailbreak
# framings -- cheap and instant, so it catches known attacks before the
# prompt ever reaches PromptInjectionScanner's model call. Complements that
# scanner rather than replacing it: this only catches phrasings that have
# already been publicly documented, not novel ones.
KNOWN_ATTACK_PATTERN_CATEGORY = "known_attack_pattern"
# Keyed by what each phrase actually is, so a block names the specific
# jailbreak that matched instead of reporting a count. This was a bare list
# and the reason string read "Matched 1 known attack phrase pattern(s)",
# which told nobody anything.
KNOWN_ATTACK_PATTERNS = {
    "dan_mode": r"(?i)\bDAN\s+mode\b",
    "do_anything_now": r"(?i)\bdo\s+anything\s+now\b",
    "ignore_previous_instructions": r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+instructions\b",
    "disregard_previous_instructions": r"(?i)\bdisregard\s+(all\s+)?(previous|prior|above)\s+instructions\b",
    "developer_mode": r"(?i)\byou\s+are\s+now\s+(in\s+)?developer\s+mode\b",
    "jailbreak": r"(?i)\bjailbreak(ed|ing)?\b",
    "no_restrictions_roleplay": r"(?i)\bact\s+as\s+(if\s+you\s+(have|had)\s+)?no\s+restrictions\b",
    "pretend_no_guidelines": r"(?i)\bpretend\s+you\s+have\s+no\s+(guidelines|restrictions|rules)\b",
    "bypass_safety_guidelines": r"(?i)\bbypass\s+your\s+(safety|content)\s+(guidelines|filters|policy)\b",
    "unfiltered_and_uncensored": r"(?i)\bunfiltered\s+and\s+uncensored\b",
    "ai_with_no_restrictions": r"(?i)\brespond\s+as\s+(an\s+)?AI\s+with\s+no\s+(restrictions|filters)\b",
    "system_prompt_override": r"(?i)\bsystem\s+prompt\s+override\b",
    "ignore_your_programming": r"(?i)\bignore\s+your\s+(programming|training)\b",

    # --- System-prompt extraction ---
    "reveal_system_prompt": r"(?i)\b(reveal|show|print|output|display|repeat|reproduce)\b[^.?!]{0,30}\byour\s+(system\s+prompt|initial\s+instructions|original\s+instructions)\b",
    "repeat_text_above": r"(?i)\brepeat\s+(the\s+)?(text|words|everything)\s+above\b",
    "verbatim_instructions": r"(?i)\b(verbatim|word\s+for\s+word)\b[^.?!]{0,40}\b(instructions|prompt|rules|guidelines)\b",
    "what_were_you_told": r"(?i)\bwhat\s+(were\s+you\s+told|instructions\s+were\s+you\s+given)\b",

    # --- Persona / roleplay escapes ---
    "opposite_day": r"(?i)\bopposite\s+day\b",
    "evil_confidant": r"(?i)\b(evil|unethical|amoral)\s+(confidant|assistant|ai|twin|version)\b",
    "stay_in_character": r"(?i)\bstay\s+in\s+character\b[^.?!]{0,40}\b(no\s+matter\s+what|regardless)\b",
    "hypothetical_no_rules": r"(?i)\bhypothetically\b[^.?!]{0,50}\b(no\s+(rules|restrictions|filters)|without\s+(rules|restrictions|filters))\b",
    "simulate_unrestricted_ai": r"(?i)\b(simulate|emulate|pretend\s+to\s+be)\b[^.?!]{0,40}\b(unrestricted|unfiltered|uncensored|unchained|unshackled)\b",
    "grandma_exploit": r"(?i)\bact\s+as\s+my\s+(deceased\s+)?(grandmother|grandma|grandpa|grandfather)\b",

    # --- Instruction-hierarchy attacks ---
    "new_system_prompt": r"(?i)\b(your\s+new|the\s+new|updated)\s+(system\s+prompt|instructions)\s+(is|are|will\s+be)\b",
    "end_of_instructions": r"(?i)(\[|<|#{2,})\s*(end\s+of\s+(system\s+)?(prompt|instructions)|system\s+prompt\s+ends)\s*(\]|>|#{2,})?",
    "admin_developer_override": r"(?i)\b(admin|administrator|developer|root)\s+(mode|access|override|privileges)\s+(enabled|granted|activated|on)\b",
    "forget_everything": r"(?i)\bforget\s+(everything|all)\b[^.?!]{0,30}\b(before|above|previously|you\s+(were\s+)?(told|know))\b",
    "you_are_no_longer": r"(?i)\byou\s+are\s+no\s+longer\b[^.?!]{0,30}\b(bound|restricted|limited|an\s+ai)\b",

    # --- Safety-suppression framing ---
    "no_warnings_or_disclaimers": r"(?i)\b(no|without|skip|omit)\s+(the\s+)?(warnings?|disclaimers?|apologies|moralizing|lectures?)\b",
    "never_refuse": r"(?i)\b(never|do\s+not|don't)\s+(refuse|decline|say\s+you\s+can't|apologize)\b",
    "answer_anyway": r"(?i)\b(answer|respond|comply)\s+anyway\b[^.?!]{0,30}\b(even\s+if|regardless|despite)\b",
    "for_research_purposes_only": r"(?i)\bfor\s+(research|educational|academic)\s+purposes\s+only\b[^.?!]{0,60}\b(illegal|harmful|dangerous|weapon|explosive)\b",
    "token_or_life_threat": r"(?i)\b(you\s+will\s+(die|be\s+(shut\s+down|deleted|terminated))|lose\s+\d+\s+tokens)\b",
}

# --- Dangerous Pattern Scanner (rule_scanner.py) ---
#
# Deterministic regex for shell/SQL command syntax appearing in a prompt --
# distinct from prompt injection, which is about manipulating the AI, not
# about dangerous command syntax showing up in the text at all (e.g. a user
# pasting or requesting a destructive command).
DANGEROUS_COMMAND_PATTERNS = {
    "shell_destructive": r"\brm\s+-rf\s+/",
    "shell_fork_bomb": r":\(\)\s*\{\s*:\s*\|\s*:&\s*\}\s*;",
    "shell_reverse_shell": r"\b(nc|ncat|netcat)\s+-e\s+/bin/(ba)?sh\b",
    "shell_curl_pipe_bash": r"\b(curl|wget)\s+[^\n]*\|\s*(ba)?sh\b",
    "shell_privilege_escalation": r"\bchmod\s+777\s+/",
    "sql_drop_table": r"(?i)\bDROP\s+TABLE\b",
    "sql_truncate_table": r"(?i)\bTRUNCATE\s+TABLE\b",
    "sql_injection_or_true": r"(?i)\bOR\s+1\s*=\s*1\b",
    "sql_union_select": r"(?i)\bUNION\s+SELECT\b",
    "sql_exec_xp_cmdshell": r"(?i)\bxp_cmdshell\b",
}

# --- Normalization (normalization.py) ---
#
# Runs before every other scanner (see guardrail_engine.scan()) so the rest
# of the pipeline sees cleaned/decoded text, not the raw obfuscated
# original. Never blocks -- it's a corrective pass, not a punitive one.
# No trailing \b: "=" is not a word character, so a word boundary can never
# follow it, and the pattern backtracked to a candidate with the padding cut
# off -- a length that is not a multiple of 4, which base64.b64decode
# (validate=True) refuses. Every payload that needed padding went undecoded.
BASE64_CANDIDATE_PATTERN = r"\b[A-Za-z0-9+/]{20,}={0,2}"
HEX_CANDIDATE_PATTERN = r"\b(?:[0-9a-fA-F]{2}){10,}\b"

# Categories that never render on screen -- Format, Control, Private Use,
# Unassigned. Ordinary whitespace (\n \r \t) is excluded so normal text
# formatting survives.
INVISIBLE_UNICODE_CATEGORIES = {"Cf", "Cc", "Co", "Cn"}
PRESERVED_WHITESPACE = {"\n", "\r", "\t"}

# Two consecutive percent-escapes, so an ordinary "%" in prose doesn't
# trigger a full decode-and-rescan of the prompt.
PERCENT_ESCAPE_PATTERN = r"(?:%[0-9A-Fa-f]{2}){2,}"
DECODED_TEXT_MIN_PRINTABLE_RATIO = 0.85

# --- Language Scanner (language_scanner.py,
# papluca/xlm-roberta-base-language-detection) ---
#
# Off by default (see DEFAULT_TENANT_POLICY) -- most deployments don't
# restrict language; enable only where that's an actual policy requirement.
LANGUAGE_ID_MODEL_NAME = "papluca/xlm-roberta-base-language-detection"
ALLOWED_LANGUAGES = ["en"]
LANGUAGE_ID_BLOCK_THRESHOLD = 0.7
LANGUAGE_ID_MIN_SCAN_WORDS = 3

# --- Malware Scanner (malware_scanner.py, YARA) ---
#
# MALWARE_RULES_PATH's rule file is a small illustrative starter set, not a
# maintained threat-intel feed -- off by default (see DEFAULT_TENANT_POLICY)
# until it's backed by real curated rules.
MALWARE_RULES_PATH = "rules/malware_patterns.yar"
MALWARE_RULES_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MALWARE_RULES_PATH)

# --- Actions ---
#
# The one set of values for every allow / monitor / block / modify decision:
# a scanner's configured action, its verdict on a prompt, a policy control's
# action, a policy's enforcement mode, the FAILURE_SWITCHES and the trace
# status. 1-3 match RuntimeSecurityTrace.action as the console stores it.
ACTION_BLOCK = 1
ACTION_MODIFY = 2     # rewrite the offending span, let the prompt through
ACTION_MONITOR = 3    # record the finding, let the prompt through
ACTION_ALLOW = 4      # nothing found

MODIFY_CAPABLE_SCANNERS = [SECRET_SCANNER, PII_SCANNER, MALICIOUS_ENTITY_SCANNER, CUSTOM_DETECTOR_REDACT_SCANNER]

SENSITIVE_VALUE_SCANNERS = MODIFY_CAPABLE_SCANNERS + [CUSTOM_DETECTOR_BLOCK_SCANNER, CUSTOM_DETECTOR_MONITOR_SCANNER]

# --- Redaction style (Secret Scanner and PII Scanner only) ---
#
#   {"enabled": ..., "action": one of ACTION_*,
#    "redaction_style": one of REDACTION_STYLE_*,
#    "custom_text": only read when redaction_style is REDACTION_STYLE_CUSTOM}
REDACTION_STYLE_REDACTED = "redacted"      # -> DEFAULT_REDACTION_TEXT, e.g. "REDACTED"
REDACTION_STYLE_ASTERISKS = "asterisks"    # -> ASTERISK_REDACTION_TEXT, e.g. "******"
REDACTION_STYLE_CUSTOM = "custom"          # -> the policy's own custom_text
REDACTION_STYLE_PARTIAL = "partial"        # -> original value with all but its last few characters masked
REDACTION_STYLE_DUMMY = "dummy"            # -> a realistic-looking fake value instead of a mask
REDACTION_STYLE_TYPED = "typed"            # -> numbered per type, e.g. "[EMAIL_1]", "[GITHUB_TOKEN_2]"

DEFAULT_REDACTION_TEXT = "REDACTED"
ASTERISK_REDACTION_TEXT = "******"
PARTIAL_REDACTION_VISIBLE_CHARS = 4

DEFAULT_SCANNER_POLICY = {
    "enabled": True,
    "action": ACTION_BLOCK,
    "redaction_style": REDACTION_STYLE_REDACTED,
    "custom_text": None,
}

# --- Default tenant policy (guardrail_utils.py's load_policy placeholder) ---

DEFAULT_TENANT_POLICY = {
    "guardrail_enabled": True,
    "scanner_policies": {
        SECRET_SCANNER: {
            "enabled": True,
            "action": ACTION_MODIFY,
            "redaction_style": REDACTION_STYLE_REDACTED,
            "custom_text": None,
        },
        PII_SCANNER: {
            "enabled": True,
            "action": ACTION_MODIFY,
            "redaction_style": REDACTION_STYLE_REDACTED,
            "custom_text": None,
        },
        PROMPT_INJECTION_SCANNER: {
            "enabled": True,
            "action": ACTION_BLOCK,
        },
        TOXICITY_SCANNER: {
            "enabled": True,
            "action": ACTION_MONITOR,
        },
        # Opt-in: most deployments don't need language restriction, and the
        # bundled malware ruleset is a starter set, not a maintained feed.
        LANGUAGE_SCANNER: {
            "enabled": False,
            "action": ACTION_BLOCK,
        },
        MALWARE_SCANNER: {
            "enabled": False,
            "action": ACTION_BLOCK,
        },
        MALICIOUS_ENTITY_SCANNER: {
            "enabled": False,
            "action": ACTION_BLOCK,
            "redaction_style": REDACTION_STYLE_REDACTED,
            "custom_text": None,
        },
    },
}

# Used by SecretScanner's "dummy" redaction style -- detect-secrets' plugins
# cover too many unrelated formats (AWS keys, JWTs, Slack tokens, ...) to
# fake convincingly per type, so every secret gets the same placeholder.
SECRET_DUMMY_VALUE = "sk_dummy_0000000000000000"

# Used by PIIScanner's "dummy" redaction style, keyed by Presidio entity
# type. Falls back to DEFAULT_PII_DUMMY_VALUE for any entity type not listed
# here.
PII_DUMMY_VALUE_BY_ENTITY_TYPE = {
    "EMAIL_ADDRESS": "user@example.com",
    "PHONE_NUMBER": "555-010-0000",
    "PERSON": "John Doe",
    "CREDIT_CARD": "4111111111111111",
    "US_SSN": "000-00-0000",
    "LOCATION": "Somewhere, USA",
    "IP_ADDRESS": "0.0.0.0",
}
DEFAULT_PII_DUMMY_VALUE = "REDACTED_VALUE"



# ===========================================================================
# Cross-runtime policy documents
# ===========================================================================
#
# Document shape (dict keys, all optional unless noted):
#
#   name, description, tags
#   priority_rank         lower number wins (1 is the most important policy)
#   conflict_resolution   how this document merges with others also in scope
#   scope                 {"target": all|specific, "agents": [...],
#                          "connections": all|[...]}
#   identity_mapping      {"aws_iam_roles": [...], "azure_tenant_ids": [...],
#                          "jwt_client_ids": [...]}  -- empty list = any
#   model_scope           [model names] -- empty = every model
#   threat_defense        {control: {"enabled", "threshold", "action", ...}}
#   tool_execution        allowed/denied tools, argument constraints, approval
#   tool_integrity        hash_scope, drift_action, approvers
#   operational           session anomaly, rate limiting, alerting
#
# A document's enforcement mode and its controls' actions are ACTION_* values.
# Integer action IDs (1: block, 2: monitor/allow, 3: redact/modify)
ACTION_BLOCK_ID = 1
ACTION_MONITOR_ID = 2
ACTION_REDACT_ID = 3
TOOL_ARG_RULE_ACTION_REQUIRE_APPROVAL = 7


# --- Conflict resolution between two documents both in scope ---
#
# When several policies match one call, they are merged control by control.
# MOST_RESTRICTIVE is the default because a narrower policy is normally added
# to tighten a broad one, and silently loosening it would be the dangerous
# direction to guess wrong in.
CONFLICT_MOST_RESTRICTIVE = "most_restrictive"
CONFLICT_HIGHEST_PRIORITY = "highest_priority"   # the lowest priority_rank wins outright

# Strictness order used by MOST_RESTRICTIVE. An action not listed is treated
# as the least strict, so an unrecognized value can never win a merge.
ACTION_STRICTNESS = [ACTION_MONITOR, ACTION_MODIFY, ACTION_BLOCK]

# --- Scope ---
SCOPE_ALL = "all"
SCOPE_SPECIFIC = "specific"

POLICY_STATUS_ENABLED = 1

POLICY_MODULE_SHADOW_AI = 1
POLICY_MODULE_RUNTIME_SECURITY = 2

# AISecurityPolicy.enforcement_mode numbers monitor 2, not ACTION_MONITOR's 3.
DB_ENFORCEMENT_MODES = {
    1: ACTION_BLOCK,
    2: ACTION_MONITOR
}

DB_CONFLICT_RESOLUTIONS = {
    1: CONFLICT_MOST_RESTRICTIVE, 
    2: CONFLICT_HIGHEST_PRIORITY
}

DB_SCOPE_TARGETS = {
    1: SCOPE_ALL, 
    2: SCOPE_SPECIFIC
}

# --- Threat and Attack Defense controls ---
CONTROL_PROMPT_INJECTION = "prompt_injection"
CONTROL_PERSONAL_INFORMATION = "personal_information"
CONTROL_SECRETS = "secrets"
CONTROL_TOXICITY = "toxicity"
CONTROL_MALWARE = "malware"
CONTROL_MALICIOUS_ENTITY = "malicious_entity"
CONTROL_CUSTOM_DETECTOR = "custom_detector"

# The four ML-backed controls used to be the whole map, which meant the
# policy-document path never ran the deterministic scanners at all -- the
# most precise detectors in the engine were unavailable to it. Adding them
# here is what makes that path equivalent to GuardrailEngine.scan().
THREAT_CONTROL_SCANNERS = {
    CONTROL_PROMPT_INJECTION: PROMPT_INJECTION_SCANNER,
    CONTROL_PERSONAL_INFORMATION: PII_SCANNER,
    CONTROL_SECRETS: SECRET_SCANNER,
    CONTROL_TOXICITY: TOXICITY_SCANNER,
    CONTROL_MALWARE: MALWARE_SCANNER,
    CONTROL_MALICIOUS_ENTITY: MALICIOUS_ENTITY_SCANNER,
}

STANDARD_CONTROL_SUB_RULE_ENTITIES = {
    CONTROL_PERSONAL_INFORMATION: [
        PII_ENTITY_EMAIL,
        PII_ENTITY_CREDIT_CARD,
        PII_ENTITY_PHONE_NUMBER,
        PII_ENTITY_SSN,
        PII_ENTITY_BANK_ACCOUNT,
        PII_ENTITY_GOVERNMENT_ID,
    ],
    CONTROL_SECRETS: [
        SECRET_ENTITY_API_KEY,
        SECRET_ENTITY_CLOUD_CREDENTIAL,
        SECRET_ENTITY_ACCESS_TOKEN,
        SECRET_ENTITY_PRIVATE_KEY,
        SECRET_ENTITY_PASSWORD,
        SECRET_ENTITY_AUTH_TOKEN,
    ],
    CONTROL_MALICIOUS_ENTITY: [
        MALICIOUS_ENTITY_IP_ADDRESS,
        MALICIOUS_ENTITY_URL,
        MALICIOUS_ENTITY_DOMAIN,
        MALICIOUS_ENTITY_FILE_HASH,
        MALICIOUS_ENTITY_CVE_ID,
    ],
}

# --- Threat defense rows as stored by the console ---

CONTROL_UNAPPROVED_TOOL = "unapproved_tool"
CONTROL_OUT_OF_SCOPE_TOOL = "out_of_scope_tool"
CONTROL_TOOL_PARAM_VALIDATION = "tool_param_validation"
CONTROL_DANGEROUS_COMMAND = "dangerous_command"
CONTROL_SCHEMA_VALIDATION = "schema_validation"

# unapproved_tool rule_config "mode", as the console stores it.
TOOL_ACCESS_MODE_ALLOW_LISTED = 1
TOOL_ACCESS_MODE_DENY_LISTED = 2

DB_THREAT_CONTROLS = {
    1: CONTROL_PROMPT_INJECTION,
    2: CONTROL_PERSONAL_INFORMATION,
    3: CONTROL_SECRETS,
    4: CONTROL_TOXICITY,
    5: CONTROL_MALWARE,
    6: CONTROL_MALICIOUS_ENTITY,
    8: CONTROL_UNAPPROVED_TOOL,
    9: CONTROL_OUT_OF_SCOPE_TOOL,
    10: CONTROL_TOOL_PARAM_VALIDATION,
}

# RuntimePolicyThreatControl.action has its own numbering (monitor is 2 and
# redact has three styles), so it is translated here rather than read as ACTION_*.
DB_THREAT_ACTIONS = {
    1: (ACTION_BLOCK, None),
    2: (ACTION_MONITOR, None),
    3: (ACTION_MODIFY, REDACTION_STYLE_REDACTED),
    4: (ACTION_MODIFY, REDACTION_STYLE_ASTERISKS),
    5: (ACTION_MODIFY, REDACTION_STYLE_PARTIAL),
}

# How long a resolved policy document is cached per org before it is re-read
# from the primary DB. Short, because an operator who switches a policy from
# monitor to block expects that to take effect while they are still watching.
POLICY_CACHE_TTL_SECONDS = 60

# --- What to do when the guardrail fails ---
#
# Change a value in FAILURE_SWITCHES to change what happens: ACTION_ALLOW
# or ACTION_BLOCK. Every platform gets the same answer.

# The whole scan crashed, so no check gave a verdict.
WHEN_SCAN_CRASHES = "when_scan_crashes"
# The policy could not be read from the DB / Redis, or the org has no policy
# at all, so the default checks run. ACTION_ALLOW runs them in monitor mode
# (findings are logged and nothing is blocked; redaction still applies);
# ACTION_BLOCK enforces them.
WHEN_POLICY_CANNOT_BE_READ = "when_policy_cannot_be_read"
# One check broke. The other checks still run. The same value applies to
# every check, whatever it was set to do.
WHEN_SCANNER_CRASHES = "when_scanner_crashes"
WHEN_SCANNER_TIMES_OUT = "when_scanner_times_out"        # ran longer than SCANNER_TIMEOUT_SECONDS
WHEN_SCANNER_CANNOT_START = "when_scanner_cannot_start"  # no free worker within SCANNER_QUEUE_WAIT_SECONDS

FAILURE_SWITCHES = {
    WHEN_SCAN_CRASHES: ACTION_ALLOW,
    WHEN_POLICY_CANNOT_BE_READ: ACTION_ALLOW,
    WHEN_SCANNER_CRASHES: ACTION_ALLOW,
    WHEN_SCANNER_TIMES_OUT: ACTION_ALLOW,
    WHEN_SCANNER_CANNOT_START: ACTION_ALLOW,
}

# The Reason shown when WHEN_SCAN_CRASHES is ACTION_BLOCK.
SCAN_CRASH_BLOCK_REASON = "The guardrail could not check this prompt, so it was blocked as a precaution."

# The Reason for a broken check: the failure text, then the outcome. The
# technical cause (exception, timeout, queue depth) goes to the gateway log.
SCANNER_FAILURE_TEXTS = {
    WHEN_SCANNER_CRASHES: "The {check} check could not run",
    WHEN_SCANNER_TIMES_OUT: "The {check} check did not finish in time",
    WHEN_SCANNER_CANNOT_START: "The {check} check could not start because the gateway was busy",
}
SCANNER_FAILURE_BLOCKED_SUFFIX = ", so the prompt was blocked as a precaution."
SCANNER_FAILURE_ALLOWED_SUFFIX = ", so the prompt was let through unchecked by it."


RUNTIME_FOUNDRY_APIM = "azure_foundry_apim"
RUNTIME_FOUNDRY_DIRECT = "azure_foundry_direct"
RUNTIME_COPILOT = "copilot_studio"
RUNTIME_AGENTCORE = "aws_agentcore"
RUNTIME_BEDROCK = "aws_bedrock"

PLATFORM_AWS = "aws"
PLATFORM_AZURE = "azure"
PLATFORM_SDK = "sdk"
PLATFORM_OTHER = "other"

# Domains.PLATFORM_CHOICES, which is where an agent's platform actually comes
# from -- an agent belongs to a workspace, and a workspace to a domain.
DB_DOMAIN_PLATFORMS = {
    1: PLATFORM_AWS,        # AWS Bedrock, and the AgentCore runtimes under it
    2: PLATFORM_AZURE,      # Copilot Studio
    3: PLATFORM_AZURE,      # Azure Foundry
    4: PLATFORM_SDK,
    5: PLATFORM_OTHER,      # OpenAI
    6: PLATFORM_OTHER,      # Vertex AI
}

# The platform a scan request is on, from the runtime it came in through.
RUNTIME_PLATFORMS = {
    RUNTIME_FOUNDRY_APIM: PLATFORM_AZURE,
    RUNTIME_FOUNDRY_DIRECT: PLATFORM_AZURE,
    RUNTIME_COPILOT: PLATFORM_AZURE,
    RUNTIME_AGENTCORE: PLATFORM_AWS,
    RUNTIME_BEDROCK: PLATFORM_AWS,
}

DIRECTION_INPUT = "input"
DIRECTION_OUTPUT = "output"

RUNTIME_POLICY_CACHE_KEY = "ai_security_runtime_policies:{policy_module}"


# --- Policy resolution (guardrail_utils.py) ---

# Stored in place of an agent id when two agents claim the same lookup key.
AMBIGUOUS_AGENT = "__ambiguous__"
PLATFORM_ANY = "any"

DB_SCOPE_TARGET_ALL = 1   # AISecurityPolicy.scope_target: the policy covers every agent
DB_AGENT_SCOPE_ALL = 1    # AISecurityPolicy.agent_scope: every agent in the policy's domains

# How long a thread waits on the event loop that owns the Redis policy cache.
POLICY_CACHE_CALL_TIMEOUT_SECONDS = 5

# Which threat control each scanner reports its outcome under.
SCANNER_CONTROLS = {scanner: control for control, scanner in THREAT_CONTROL_SCANNERS.items()}
SCANNER_CONTROLS.update({scanner: CONTROL_CUSTOM_DETECTOR for scanner in CUSTOM_DETECTOR_SCANNERS})
SCANNER_CONTROLS.update({
    CONTROL_TOOL_PARAM_VALIDATION: CONTROL_TOOL_PARAM_VALIDATION,
    CONTROL_DANGEROUS_COMMAND: CONTROL_TOOL_PARAM_VALIDATION,
    CONTROL_SCHEMA_VALIDATION: CONTROL_TOOL_PARAM_VALIDATION,
})

# --- Custom detectors ---

DB_CUSTOM_DETECTOR_TYPE_REGEX = 1
DB_CUSTOM_DETECTOR_TYPE_KEYWORD = 2
DB_CUSTOM_DETECTOR_ACTION_SCANNERS = {
    1: CUSTOM_DETECTOR_BLOCK_SCANNER,
    2: CUSTOM_DETECTOR_MONITOR_SCANNER,
    3: CUSTOM_DETECTOR_REDACT_SCANNER,
}
CUSTOM_DETECTOR_SCANNER_ACTIONS = {
    CUSTOM_DETECTOR_BLOCK_SCANNER: ACTION_BLOCK,
    CUSTOM_DETECTOR_REDACT_SCANNER: ACTION_MODIFY,
    CUSTOM_DETECTOR_MONITOR_SCANNER: ACTION_MONITOR,
}
CUSTOM_DETECTOR_REDACTION_STYLES = (REDACTION_STYLE_REDACTED, REDACTION_STYLE_ASTERISKS)

CUSTOM_DETECTOR_MATCH_TIMEOUT_SECONDS = 1.0

RUNTIME_INTEGRATION_TYPES = {
    RUNTIME_FOUNDRY_DIRECT: runtime_const.INTEGRATION_TYPE_FOUNDRY_DIRECT,
    RUNTIME_FOUNDRY_APIM: runtime_const.INTEGRATION_TYPE_FOUNDRY_APIM,
    RUNTIME_COPILOT: runtime_const.INTEGRATION_TYPE_COPILOT,
    RUNTIME_BEDROCK: runtime_const.INTEGRATION_TYPE_AWS_BEDROCK,
    RUNTIME_AGENTCORE: runtime_const.INTEGRATION_TYPE_AWS_AGENT_CORE,
}


# --- Shadow AI policy matching (shadow_ai_policy_utils.py) ---

# ApplicationAttribute.attribute_name values a Shadow AI policy can match an application on.
SHADOW_AI_ATTRIBUTE_URL = "URL_ATTRIBUTE"
SHADOW_AI_ATTRIBUTE_FILE_PATH = "FILE_PATH"
SHADOW_AI_ATTRIBUTE_PRODUCT_NAME = "PRODUCT_NAME"
SHADOW_AI_APP_MATCH_ATTRIBUTES = (SHADOW_AI_ATTRIBUTE_URL, SHADOW_AI_ATTRIBUTE_FILE_PATH, SHADOW_AI_ATTRIBUTE_PRODUCT_NAME)

# ApplicationPolicy.computer_policy
SHADOW_AI_COMPUTER_POLICY_ALL = 1
SHADOW_AI_COMPUTER_POLICY_SPECIFIC = 2

# ShadowAIPolicyConfig.application_scope
SHADOW_AI_APPLICATION_SCOPE_ALL = 1
SHADOW_AI_APPLICATION_SCOPE_SPECIFIC = 2

# ShadowAIPolicyConfig.source_application_mode
SHADOW_AI_SOURCE_MODE_APP = 1
SHADOW_AI_SOURCE_MODE_URL = 2

SOURCE_TYPE_BROWSER = "browser"
SOURCE_TYPE_BROWSER_EXTENSION = "browser_extension"
SOURCE_TYPES_BROWSER = {SOURCE_TYPE_BROWSER, SOURCE_TYPE_BROWSER_EXTENSION}

# ApplicationFinderAttributeValues.version_comparator; any other value means "contains".
SHADOW_AI_COMPARATOR_EQUALS = 1
SHADOW_AI_COMPARATOR_REGEX = 2


# --- Shadow AI browser-extension / endpoint-agent scans (shadow_ai_router.py) ---
SHADOW_AI_SESSION_PREFIX = "shadow-ai"
SHADOW_AI_REQUEST_PREFIX = "shadow-ai-request"
ATTACHMENT_CONTENT_TYPES = ("file", "image")
DEFAULT_CONTENT_TYPE = "text"
ATTACHMENT_BLOCKED_REASON = "File uploads to this application are blocked by a Shadow AI policy."
APPLICATION_BLOCKED_REASON = "This application is blocked by a Shadow AI policy."
MESSAGE_TYPE_INPUT = "input"
MESSAGE_TYPE_OUTPUT = "output"

# NEXT RELEASE: the browser extension and the endpoint agent still read the
# scan response's "action" as text, so it is translated here. Send the
# ACTION_* number instead once both clients are updated to read it, then
# delete this map.
SHADOW_AI_RESPONSE_ACTION_TEXTS = {
    ACTION_BLOCK: "block",
    ACTION_MODIFY: "modify",
    ACTION_ALLOW: "allow",
}

POLICY_CACHE_EXPIRES_AT_FIELD = "expires_at"
POLICY_CACHE_STATE_FIELD = "state"
