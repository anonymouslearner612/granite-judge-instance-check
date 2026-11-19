import os
from pathlib import Path

INTEGRATION_TYPE_FOUNDRY_DIRECT = 1
INTEGRATION_TYPE_FOUNDRY_APIM = 2
INTEGRATION_TYPE_COPILOT = 3
INTEGRATION_TYPE_AWS_BEDROCK = 4
INTEGRATION_TYPE_AWS_AGENT_CORE = 5

# Timeouts (seconds) for the direct proxy's upstream call to the model.
# httpx defaults to 5 seconds for every phase, which is far below the time a
# model can take to produce its first streamed token.
LLM_PROXY_CONNECT_TIMEOUT_SECONDS = 15.0
LLM_PROXY_READ_TIMEOUT_SECONDS = 300.0
LLM_PROXY_WRITE_TIMEOUT_SECONDS = 60.0
LLM_PROXY_POOL_TIMEOUT_SECONDS = 15.0
DIRECT_PROXY_SCAN_TIMEOUT_SECONDS = 120.0
MAX_DIRECT_PROXY_BUFFER_SIZE = 10 * 1024 * 1024  # 10 MB maximum buffer for upstream responses
UPLOAD_INTERVAL_MINUTES = 10
TEMP_DIR = str(Path(__file__).resolve().parents[3] / "temp")
RUNTIME_TRACES_DIR = os.path.join(TEMP_DIR, "runtime_traces")

DIRECTION_INBOUND = 1
DIRECTION_OUTBOUND = 2

STATUS_CODE_OK = 200
STATUS_CODE_BAD_REQUEST = 400
STATUS_CODE_FORBIDDEN = 403
STATUS_CODE_BAD_GATEWAY = 502

# --- Runtime router ---
APIM_REQUEST_TYPE_REQUEST = "request"
APIM_REQUEST_TYPE_RESPONSE = "response"

RESPONSE_STATUS_BYPASS = "bypass"
RESPONSE_STATUS_PARSE_ERROR_ALLOWED = "parse_error_allowed"
RESPONSE_STATUS_UNKNOWN_TYPE_ALLOWED = "unknown_type_allowed"

TRACE_TYPE_REQUEST = "request"
TRACE_TYPE_COPILOT_TOOL_EXECUTION = "copilot_tool_execution"

REDTEAM_BLOCK_REASON = "Blocked by Securden Redteam Interception"
SECURITY_POLICY_BLOCK_MESSAGE = "Blocked by Securden: Security policy violation"
HISTORY_BLOCKED_MESSAGE_PLACEHOLDER = "[Removed by Security Policy]"
THREAT_DETECTION_BLOCK_REASON = "Blocked by Securden threat detection."
THREAT_DETECTION_MONITOR_REASON = "Flagged by Securden threat detection and let through."
GATEWAY_ALLOW_REASON = "Allowed by Securden Gateway"
AGENT_NOT_OPTED_IN_REASON = "Agent not opted in for scanning"

# Copilot Studio's threat-detection webhook reads these reason codes.
COPILOT_REASON_CODE_BLOCKED = 1001
COPILOT_REASON_CODE_ALLOWED = 0
CORRELATION_ID_HEADER = "x-ms-correlation-id"

RUNTIME_API_KEY_HEADER = "x-api-key"
AZURE_API_KEY_HEADER = "api-key"
CHAT_COMPLETIONS_PATH_MARKER = "chat/completions"
FOUNDRY_MODELS_URL = "https://{hub_name}.services.ai.azure.com/models"
DIRECT_PROXY_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
# Methods Azure Foundry uses to check the endpoint is reachable, never to send traffic.
DIRECT_PROXY_PING_METHODS = ["GET", "OPTIONS"]
EVENT_STREAM_MEDIA_TYPE = "text/event-stream"
JSON_MEDIA_TYPE = "application/json"
FOUNDRY_PROXY_ALLOWED_HEADERS = frozenset({
    "content-type",
    "accept",
    "traceparent",
    "tracestate",
    "x-ms-client-request-id",
    "x-ms-client-session-id",
    "x-ms-correlation-id",
    "x-ms-useragent",
    "x-ms-foundry-agent-id",
    "x-request-id",
    "user-agent",
})

# --- Payload parsing (runtime_utils.py) ---
SSE_DATA_PREFIX = "data: "
SSE_DONE_LINE = "data: [DONE]"
ROLE_USER = "user"
CONTENT_TYPE_TEXT = "text"
NO_USER_TEXT_PLACEHOLDER = "<No direct user text found>"
UNKNOWN_USER = "Unknown User"
FOUNDRY_AGENT_ID_HEADER = "x-ms-foundry-agent-id"
# Foundry sends this in place of an agent id for traffic no agent made.
FOUNDRY_NO_AGENT_VALUE = "n/a"

# --- Trace capture (runtime_trace_capture.py, shadow_ai_trace_capture.py) ---
SHADOW_AI_TRACES_DIR = os.path.join(TEMP_DIR, "shadow_ai_traces")
TRACE_FILE_EXTENSION = ".json"
AWS_UPLOAD_KEY = "aws_upload_key"
RUNTIME_TRACES_S3_FOLDER = "traces/runtime_traces/{org_id}/{agent_name}/{session_id}"
SHADOW_AI_TRACES_S3_FOLDER = "traces/shadow_ai_traces/{org_id}/{computer_id}/{session_id}"
UNMAPPED_AGENT_FOLDER = "unmapped"
TRACE_ID_PATTERN = r"[A-Za-z0-9_-]{1,128}"

CLIENT_SESSION_ID_HEADER = "x-ms-client-session-id"
CLIENT_REQUEST_ID_HEADER = "x-ms-client-request-id"
TRACEPARENT_HEADER = "traceparent"


UPSTREAM_FAILURE_MESSAGE = "The model did not respond in time. Please retry."

SCAN_SUMMARY_BATCH_SIZE = 10
SCAN_SUMMARY_REDIS_EXPIRY_SECONDS = 86400
SCAN_SUMMARY_SHADOW_AI_KEY_TAG = "shadow_ai"

PROMPT_SCAN_CACHE_TTL_SECONDS = 86400
PROMPT_SCAN_CACHE_KEY_TEMPLATE = "runtime:prompt_scan:{agent_name}:{prompt_hash}"

# Redis hash fields that break blocked_count and redacted_count down. Each
# blocked scan adds 1 to exactly one "blocked:<reason>" field and each
# redacted scan to exactly one "redacted:<ACTION_MODIFY or ACTION_MONITOR>"
# field, in the same pipeline as the totals, so the breakdowns always add up
# to their counts.
SCAN_SUMMARY_BLOCKED_FIELD_PREFIX = "blocked:"
SCAN_SUMMARY_REDACTED_FIELD_PREFIX = "redacted:"
# Block reason when no scanner decided it: a Shadow AI application policy
# blocks before any scan, anything else is unclassified.
SCAN_SUMMARY_REASON_SHADOW_POLICY = "shadow_ai_policy"
SCAN_SUMMARY_REASON_UNCLASSIFIED = "unclassified"
# Where a Shadow AI scan came from, kept on the summary row.
SCAN_SUMMARY_SOURCE_BROWSER = "browser_extension"
SCAN_SUMMARY_SOURCE_ENDPOINT = "endpoint"
SCAN_SUMMARY_RUNTIME_KEY_TEMPLATE = "scan_summary:{agent_id}:{hour}"
SCAN_SUMMARY_SHADOW_AI_KEY_TEMPLATE = "scan_summary:" + SCAN_SUMMARY_SHADOW_AI_KEY_TAG + ":{source_type}:{computer_id}:{hour}"
# The org's index of counter keys written in an hour, so the next hour can save their leftovers.
SCAN_SUMMARY_ACTIVE_SET_KEY_TEMPLATE = "active_scan_summaries:{hour}"

ERROR_BEHAVIOR_ALLOW = 1
ERROR_BEHAVIOR_BLOCK = 2
