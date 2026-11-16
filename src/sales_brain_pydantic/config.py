from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Mapping


_ENV_KEYS = {
    "BRAIN_API_TOKEN", "BRAIN_STATE_DIR", "OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_MODE",
    "OPENAI_MODEL", "OPENAI_REASONING_EFFORT", "OPENAI_MAX_OUTPUT_TOKENS",
    "BRAIN_TOTAL_TIMEOUT_SECONDS", "BRAIN_MODEL_REQUEST_LIMIT", "BRAIN_TOOL_CALL_LIMIT",
    "SA_SERVICE_REGISTRY_FILE", "SA_MCP_REGISTRY_FILE", "SA_TENANT_ID",
    "KNOWLEDGE_API_TOKEN", "KNOWLEDGE_BASE_URL", "KNOWLEDGE_MODEL",
}


class Settings:
    """Small settings container that accepts private JSON and environment overrides."""

    def __init__(self, values: Mapping[str, object] | None = None):
        self.values = {str(key): value for key, value in (values or {}).items()}
        self._validate()

    @classmethod
    def load(cls, *, path: str | Path | None = None,
             environ: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if environ is None else environ
        config_path = Path(path or env.get("BRAIN_CONFIG_FILE", "./service.json"))
        raw: dict[str, object] = {}
        if config_path.exists():
            info = config_path.lstat()
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise ValueError("brain_config_must_be_regular_file")
            if info.st_mode & 0o007 or info.st_mode & 0o020:
                raise ValueError("brain_config_permissions_too_open")
            parsed = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(parsed, dict):
                raise ValueError("brain_config_must_be_object")
            raw.update(parsed)
        for key in _ENV_KEYS:
            if key in env:
                raw[key] = env[key]
        return cls(raw)

    def _validate(self) -> None:
        self._set_default("OPENAI_MODEL", "example-model")
        self._set_default("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self._set_default("OPENAI_API_MODE", "responses")
        self._set_default("OPENAI_REASONING_EFFORT", "medium")
        self._set_default("OPENAI_MAX_OUTPUT_TOKENS", 8192)
        self._set_default("BRAIN_TOTAL_TIMEOUT_SECONDS", 240)
        self._set_default("BRAIN_MODEL_REQUEST_LIMIT", 12)
        self._set_default("BRAIN_TOOL_CALL_LIMIT", 24)
        self._set_default("BRAIN_STATE_DIR", "./var")
        for key, lower, upper in (
            ("OPENAI_MAX_OUTPUT_TOKENS", 1024, 32768),
            ("BRAIN_TOTAL_TIMEOUT_SECONDS", 30, 240),
            ("BRAIN_MODEL_REQUEST_LIMIT", 1, 12),
            ("BRAIN_TOOL_CALL_LIMIT", 1, 24),
        ):
            try:
                value = int(self.values[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid_{key.lower()}") from exc
            if not lower <= value <= upper:
                raise ValueError(f"invalid_{key.lower()}")
            self.values[key] = value
        if self.values["OPENAI_REASONING_EFFORT"] not in {"none", "low", "medium", "high", "xhigh"}:
            raise ValueError("invalid_openai_reasoning_effort")
        if self.values["OPENAI_API_MODE"] not in {"responses", "chat"}:
            raise ValueError("invalid_openai_api_mode")
        if self.values["OPENAI_API_MODE"] == "chat" and self.values["OPENAI_REASONING_EFFORT"] != "none":
            raise ValueError("chat_api_requires_reasoning_effort_none")
        if not isinstance(self.values["OPENAI_MODEL"], str) or not self.values["OPENAI_MODEL"].strip():
            raise ValueError("invalid_openai_model")
        if not isinstance(self.values["OPENAI_BASE_URL"], str) or not self.values["OPENAI_BASE_URL"].startswith(("https://", "http://localhost", "http://127.0.0.1")):
            raise ValueError("invalid_openai_base_url")
        state_dir = self.values["BRAIN_STATE_DIR"]
        if not isinstance(state_dir, str) or not state_dir.strip():
            raise ValueError("invalid_brain_state_dir")

    def _set_default(self, key: str, value: object) -> None:
        if key not in self.values:
            self.values[key] = value

    def get(self, key: str, default: object = None) -> object:
        return self.values.get(key, default)

    def require(self, key: str) -> str:
        value = self.values.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing_{key.lower()}")
        return value.strip()

    @property
    def database_path(self) -> Path:
        return Path(str(self.values["BRAIN_STATE_DIR"])) / "sales-brain-pydantic-public.sqlite3"

    @property
    def version(self) -> str:
        return "0.1.0"

