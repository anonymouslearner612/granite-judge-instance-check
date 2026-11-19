# Pydantic is used here as a deliberate exception to the root CLAUDE.md's
# project-wide "no Pydantic" rule -- these are pure data shapes, and Pydantic
# validates them for free. Every other class in this package is a plain class.

from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import Field

from app.guardrail import constants as guardrail_const
from app.schemas import base_model


class ScannerResult(BaseModel):
    scanner_name: str
    detected: bool = False
    categories: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    risk_score: float = 0.0
    recommendation: int = guardrail_const.ACTION_ALLOW
    modified_prompt: Optional[str] = None
    hidden_content_text: Optional[str] = None
    sensitive_values: List[str] = Field(default_factory=list, exclude=True, repr=False)
    redactions: List[dict] = Field(default_factory=list, exclude=True, repr=False)
    explanation: str = ""
    execution_time_seconds: float = 0.0
    succeeded: bool = True
    timed_out: bool = False


class GuardrailResponse(BaseModel):
    is_allowed: bool
    is_blocked: bool
    is_modified: bool
    prompt: str
    risk_score: Optional[float] = None
    categories: Optional[List[str]] = None
    control_outcomes: Dict[str, int] = Field(default_factory=dict)
    reason: Optional[str] = None
    is_clean_prompt: bool = True
    # Every scanner's own verdict, in execution order. Without this a caller
    # only sees a merged risk score and a joined reason string, so it can't
    # tell which scanner actually decided the outcome.
    scanner_results: List[ScannerResult] = Field(default_factory=list)
    failed_scanner_names: List[str] = Field(default_factory=list)
    deciding_scanner: Optional[str] = None
    policy_id: Optional[int] = None
    policy_name: Optional[str] = None
    sensitive_values: List[str] = Field(default_factory=list, exclude=True, repr=False)


# --- API-layer schemas -----------------------------------------------------
#
# Request/response shapes for the FastAPI routers (shadow_ai_router) and
# their callers (security/constraints.py).
# StrictBaseModel rejects unknown fields, matching the rest of the app's API
# schemas.


class Message(base_model.StrictBaseModel):
    role: str = "user"
    content: str


class ChatCompletionRequest(base_model.StrictBaseModel):
    messages: List[Message]
    agent_instruction: str = ""
    prompt_typed_at: str = ""
    hostname: str = ""
    source_ip: str = ""
    url: str = ""
    type: str = "text"
    content_source: str = ""
    source_type: str = ""
    user_sid: str = ""
    application_type: str = guardrail_const.APPLICATION_TYPE_AGENT
    mcp_enabled: bool = False
    username: str = ""
    browser: str = ""
    extension_id: str = ""

    request_id: str = ""
    session_id: str = ""
    device_id: str = ""
    application: str = ""
    application_path: str = ""
    # Client-reported matches, accepted for compatibility and logging only.
    # The gateway resolves the effective policy from the computer and app/URL.
    matched_app_policy_id: Optional[int] = None
    matched_policy_id: Optional[int] = None
    matched_policy_ids: List[int] = Field(default_factory=list)
    model: Optional[str] = None
    provider: Optional[str] = None
    conversation_id: Optional[str] = None
    # Logged only -- which scanners run is decided by the matched policy, not this list.
    policies: List[str] = Field(default_factory=list)


class InternalScanRequest(base_model.StrictBaseModel):
    prompt: str
    # The primary server decides which scanners run and how; a scanner left out is not run.
    scanner_policies: Dict[str, dict] = Field(default_factory=dict)


class PromptResponse(base_model.StrictBaseModel):
    """
    NEXT RELEASE: action is still text because the browser extension and the
    endpoint agent read it that way. Switch it to the ACTION_* number together
    with both clients (see SHADOW_AI_RESPONSE_ACTION_TEXTS).
    """
    is_allowed: bool
    is_blocked: bool
    is_modified: bool
    is_monitored: bool = False
    redactions: List[dict] = Field(default_factory=list)
    prompt: str
    risk_score: Optional[float] = None
    categories: Optional[List[str]] = None
    reason: Optional[str] = None
    # "block", "modify", or "allow" -- modify means `prompt` carries the redacted text.
    # Monitor decisions keep action="allow" for older clients and set is_monitored=True.
    action: str = guardrail_const.SHADOW_AI_RESPONSE_ACTION_TEXTS[guardrail_const.ACTION_ALLOW]
