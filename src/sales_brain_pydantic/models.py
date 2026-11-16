from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class HistoryMessage(StrictModel):
    role: Literal["user", "assistant", "participant"]
    original_speaker_type: str = Field(default="", max_length=80)
    text: str = Field(min_length=1)
    time_beijing: str = Field(default="", max_length=80)


class ChatRequest(StrictModel):
    request_id: str = Field(min_length=8, max_length=120)
    customer_id: str | None = Field(default=None, min_length=1, max_length=160)
    sales_region: str | None = Field(default=None, min_length=2, max_length=32)
    reply_language: str = Field(default="zh-CN", min_length=2, max_length=80)
    conversation_time_beijing: datetime | None
    history: list[HistoryMessage] = Field(default_factory=list, max_length=500)
    message: str = Field(min_length=1)

    @field_validator("conversation_time_beijing", mode="before")
    @classmethod
    def normalize_beijing_time(cls, value: datetime | str | None) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        if value.tzinfo is None:
            return value.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
        return value.astimezone(ZoneInfo("Asia/Shanghai"))

    @model_validator(mode="after")
    def normalize_region(self) -> "ChatRequest":
        if self.sales_region:
            raw = self.sales_region.strip().upper().replace(" ", "")
            aliases = {"REGION-A": "REGION-A", "REGIONA": "REGION-A", "REGION-B": "REGION-B", "REGIONB": "REGION-B"}
            if raw not in aliases:
                raise ValueError("unsupported_sales_region")
            self.sales_region = aliases[raw]
        return self


ActionStatus = Literal["proposed", "needs_confirmation", "blocked"]


class SuggestedAction(StrictModel):
    kind: Literal[
        "search_products", "send_product_media", "quote_request", "sample_request",
        "order_request", "payment_check", "create_followup", "human_handoff",
        "ask_clarification", "no_action",
    ]
    summary: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list, max_length=32)
    preconditions: list[str] = Field(default_factory=list, max_length=32)
    status: ActionStatus = "proposed"
    execution_status: Literal["not_executed"] = "not_executed"


class SalesDecision(StrictModel):
    reply: str
    actions: list[SuggestedAction] = Field(default_factory=list, max_length=32)
    evidence_refs: list[str] = Field(default_factory=list, max_length=64)
    missing_information: list[str] = Field(default_factory=list, max_length=64)
    status: Literal["ready", "needs_information", "blocked", "unavailable"]


class UsageSummary(StrictModel):
    requests: int = 0
    tool_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class ChatResponse(StrictModel):
    run_id: str
    request_id: str
    version: str
    status: Literal["planning", "ready", "needs_information", "blocked", "unavailable", "failed", "stale", "cancelled"]
    reply: str = ""
    actions: list[SuggestedAction] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    usage: UsageSummary = Field(default_factory=UsageSummary)
    tool_trace: list[dict[str, Any]] = Field(default_factory=list)
    model_messages: list[dict[str, Any]] = Field(default_factory=list)
    execution_receipts: list[dict[str, Any]] = Field(default_factory=list)
    error: dict[str, str] | None = None

