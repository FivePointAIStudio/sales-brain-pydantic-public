"""Read-only business integrations for the PydanticAI sales brain.

Credentials and endpoint registries are read from private files. Tool arguments
never select a customer, tenant, region, URL, SQL statement, or credential.
"""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
import hashlib
import ipaddress
import json
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
import socket
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
from urllib.request import (HTTPRedirectHandler, ProxyHandler, Request,
                            build_opener)
import uuid


REGIONS = ("REGION-A", "REGION-B")
HTTP_SERVICES = {
    "customer-context", "product-catalog", "product-matching", "pricing",
    "company-knowledge", "sales-workflows",
}
READONLY_MCP_TOOLS = frozenset({"business_guide", "validate_readonly_sql", "mysql_query",
    "get_query_kit", "read_dictionary_doc", "search_data_dictionary", "search_lessons",
    "resolve_intent", "get_doc_atom"})
MAX_BODY = 2_000_000
DEFAULT_TIMEOUT = 12.0
_SAFE_SQL_LITERAL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_TRANSPORT_TRACE: ContextVar[list[dict[str, Any]] | None] = ContextVar("business_transport_trace", default=None)


class IntegrationError(RuntimeError):
    """Safe integration error; its message never includes remote payloads."""


def _trace_safe(value: Any, key: str = "", depth: int = 0) -> Any:
    """Remove credentials and bound the private protocol trace payload."""
    low = key.lower()
    if any(part in low for part in ("token", "secret", "password", "authorization", "api_key", "credential")):
        return "[redacted]"
    if depth > 8:
        return "[depth_limit]"
    if isinstance(value, Mapping):
        return {str(k): _trace_safe(v, str(k), depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_trace_safe(v, key, depth + 1) for v in value[:1000]]
    if isinstance(value, str):
        if "url" in low or "uri" in low:
            parts = urlsplit(value)
            if parts.scheme and parts.netloc:
                return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
        return value[:30000]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:1000]


def _trace_event(event: Mapping[str, Any]) -> None:
    trace = _TRANSPORT_TRACE.get()
    if trace is None:
        return
    safe = _trace_safe(event)
    try:
        if len(json.dumps(safe, ensure_ascii=False, separators=(",", ":"))) > 100_000:
            safe = {**safe, "response": "[payload_too_large]", "trace_truncated": True}
    except (TypeError, ValueError):
        safe = {"transport": event.get("transport"), "status": "trace_serialization_error"}
    trace.append(dict(safe))


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def _get(settings: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(settings, Mapping) and name in settings:
            return settings[name]
        values = getattr(settings, "values", None)
        if isinstance(values, Mapping):
            if name in values:
                return values[name]
            if name.upper() in values:
                return values[name.upper()]
        getter = getattr(settings, "get", None)
        if callable(getter):
            for key in (name, name.upper()):
                try:
                    value = getter(key)
                except (TypeError, KeyError):
                    continue
                if value is not None:
                    return value
        if hasattr(settings, name):
            value = getattr(settings, name)
            if value is not None:
                return value
    return default


def _json_file(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise IntegrationError("private_config_unavailable")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise IntegrationError("private_config_invalid") from None
    if not isinstance(value, dict):
        raise IntegrationError("private_config_invalid")
    return value


def _read_settings(settings: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return (service registry, MCP registry, private service settings)."""
    private = _get(settings, "private_config", "private_settings", default=None)
    values = getattr(settings, "values", None)
    if private is None and isinstance(values, Mapping):
        private = values
    if private is None and isinstance(settings, Mapping):
        private = settings
    if private is None:
        private = {}
    if isinstance(private, (str, Path)):
        private = _json_file(private)
    if not isinstance(private, Mapping):
        private = {}
    secret_path = (_get(settings, "private_config_file", "service_config_file", "secrets_file", "BRAIN_CONFIG_FILE")
                   or os.environ.get("BRAIN_CONFIG_FILE"))
    if secret_path:
        private = {**_json_file(secret_path), **private}
    service_path = (_get(settings, "service_registry_file", "service_registry_path")
                    or private.get("SA_SERVICE_REGISTRY_FILE")
                    or os.environ.get("SA_SERVICE_REGISTRY_FILE"))
    mcp_path = (_get(settings, "mcp_registry_file", "mcp_registry_path")
                or private.get("SA_MCP_REGISTRY_FILE")
                or os.environ.get("SA_MCP_REGISTRY_FILE"))
    service_registry = _get(settings, "service_registry", default=None)
    if service_registry is None:
        service_registry = _json_file(service_path)
    mcp_registry = _get(settings, "mcp_registry", default=None)
    if mcp_registry is None:
        mcp_registry = _json_file(mcp_path)
    return (dict(service_registry or {}), dict(mcp_registry or {}), dict(private))


def _schema_obj(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"type": "object", "properties": properties,
            "required": list(required), "additionalProperties": False}


_STR = {"type": "string", "minLength": 1}
_PRODUCT_REF = {"type": "string", "minLength": 1, "maxLength": 160}
_DATE = {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {"name": "read_customer_context", "description": "Read this already resolved customer's profile and only conversation history at or before the request time.", "parameters": _schema_obj({})},
    {"name": "resolve_product_reference", "description": "Resolve an exact product code in the trusted sales region.", "parameters": _schema_obj({"reference": {"type": "string", "minLength": 1, "maxLength": 128}}, ("reference",))},
    {"name": "search_products", "description": "Search current public catalog candidates in the trusted customer and region scope.", "parameters": _schema_obj({"query": {"type": "string", "minLength": 1, "maxLength": 3000}, "conditions": {"type": "array", "maxItems": 16, "items": _schema_obj({"field": {"type": "string", "maxLength": 64}, "operator": {"type": "string", "enum": ["eq", "gte", "lte", "contains", "in"]}, "value": {}, "quote": {"type": "string", "maxLength": 2000}}, ("field", "operator", "value"))}, "limit": {"type": "integer", "minimum": 1, "maximum": 10}}, ("query",))},
    {"name": "get_product_media", "description": "Read media for a product previously verified in this run.", "parameters": _schema_obj({"product_ref": _PRODUCT_REF}, ("product_ref",))},
    {"name": "resolve_quote", "description": "Read current applicable quote facts for products previously resolved in this run. Missing quantity or terms remain unknown.", "parameters": _schema_obj({"product_refs": {"type": "array", "items": _PRODUCT_REF, "minItems": 1, "maxItems": 10, "uniqueItems": True}, "quantity": {"type": "number", "exclusiveMinimum": 0}, "unit": {"type": "string", "enum": ["m", "y", "kg", "Pcs"]}, "currency": {"type": "string", "pattern": "^[A-Z]{3}$"}}, ("product_refs",))},
    {"name": "search_company_knowledge", "description": "Search published public company documents; this does not verify current price, stock, order, or delivery facts.", "parameters": _schema_obj({"question": {"type": "string", "minLength": 1, "maxLength": 1000}}, ("question",))},
    {"name": "query_knowledge", "description": "Ask the configured secondary knowledge model. Its answer is reference material, not verified business evidence.", "parameters": _schema_obj({"question": {"type": "string", "minLength": 1, "maxLength": 4000}}, ("question",))},
    {"name": "erp_sample_card_lookup", "description": "Read this confirmed customer's ERP sample-card application records; requires exact product or sample-card identifier.", "parameters": _schema_obj({"question": {"type": "string", "minLength": 1, "maxLength": 1000}, "goods_no": {"type": "string", "minLength": 1, "maxLength": 128}, "sample_card_no": {"type": "string", "minLength": 1, "maxLength": 128}}, ("question",))},
    {"name": "erp_order_progress", "description": "Read the trusted customer's ERP order header status by exact bill number. It does not verify shipment or receipt.", "parameters": _schema_obj({"question": {"type": "string", "minLength": 1, "maxLength": 1000}, "bill_no": {"type": "string", "minLength": 1, "maxLength": 128}, "intent_ref": {"type": "string", "minLength": 1, "maxLength": 240}}, ("question",))},
    {"name": "erp_customer_debt", "description": "Read the confirmed customer's current-month RMB receivable snapshot.", "parameters": _schema_obj({"question": {"type": "string", "minLength": 1, "maxLength": 1000}}, ("question",))},
    {"name": "erp_product_facts", "description": "Read public current product specification facts for catalog-verified products.", "parameters": _schema_obj({"product_refs": {"type": "array", "items": _PRODUCT_REF, "minItems": 1, "maxItems": 10, "uniqueItems": True}}, ("product_refs",))},
    {"name": "erp_customer_activity", "description": "Read bounded customer-owned order, sample application, logistics, or refund records. Filters narrow a read and do not establish transaction selection.", "parameters": _schema_obj({"topic": {"type": "string", "enum": ["orders", "samples", "logistics", "refunds"]}, "goods_no": {"type": "string", "maxLength": 128}, "bill_no": {"type": "string", "maxLength": 128}, "color_no": {"type": "string", "maxLength": 128}, "date_start": _DATE, "date_end": _DATE, "order_kind": {"type": "string", "enum": ["meter_sample", "bulk"]}, "limit": {"type": "integer", "minimum": 1, "maximum": 30}}, ("topic",))},
    {"name": "erp_inventory_snapshot", "description": "Read bounded inventory records for a catalog-verified product in the trusted region. Empty rows do not mean zero stock; records do not reserve or guarantee sale/shipment.", "parameters": _schema_obj({"product_ref": _PRODUCT_REF, "color_codes": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 64}, "minItems": 1, "maxItems": 20}, "source_event_refs": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 240}, "minItems": 1, "maxItems": 20}}, ("product_ref",))},
    {"name": "read_workflow_context", "description": "Read this customer's existing sales cases and requests without creating or changing them.", "parameters": _schema_obj({})},
    {"name": "read_sales_capabilities", "description": "Read current workflow capabilities and availability; this tool does not execute them.", "parameters": _schema_obj({})},
    {"name": "read_sales_owner", "description": "Read the assigned salesperson identity for this customer and region.", "parameters": _schema_obj({})},
]


def _validate(schema: Mapping[str, Any], value: Any, path: str = "args") -> None:
    if schema.get("type") == "object":
        if not isinstance(value, Mapping):
            raise ValueError(path + "_object_required")
        props = schema.get("properties", {})
        if set(value) - set(props):
            raise ValueError(path + "_unknown_field")
        if set(schema.get("required", ())) - set(value):
            raise ValueError(path + "_required_field_missing")
        for key, item in value.items():
            _validate(props[key], item, path + "_" + key)
        return
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(path + "_invalid_value")
    kind = schema.get("type")
    if kind == "string":
        if not isinstance(value, str) or len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", 100000):
            raise ValueError(path + "_invalid_string")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise ValueError(path + "_invalid_string")
    elif kind == "integer":
        if type(value) is not int or value < schema.get("minimum", -10**18) or value > schema.get("maximum", 10**18):
            raise ValueError(path + "_invalid_integer")
    elif kind == "number":
        if isinstance(value, bool) or not isinstance(value, (float, int)) or value <= schema.get("exclusiveMinimum", float("-inf")):
            raise ValueError(path + "_invalid_number")
    elif kind == "array":
        if not isinstance(value, list) or len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", 1000):
            raise ValueError(path + "_invalid_array")
        if schema.get("uniqueItems") and len(value) != len(set(value)):
            raise ValueError(path + "_duplicate_items")
        for item in value:
            _validate(schema["items"], item, path + "_item")


def _hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _sql_literal(value: Any) -> str:
    """Return a strictly bounded ERP literal, or fail before SQL construction."""
    if not isinstance(value, str):
        raise IntegrationError("business_identifier_invalid")
    text = value.strip()
    if not text or not _SAFE_SQL_LITERAL.fullmatch(text) or any(ord(c) < 32 for c in text):
        raise IntegrationError("business_identifier_invalid")
    return text.replace("'", "''")


def _inventory_dictionary_digest(text: str, source_system: str) -> str | None:
    """Return the live stock-guide-v1 body hash only when its stock contract is present."""
    header = "===== example_analytics/stock-query-guide.md ====="
    if source_system != "configured_local_mcp" or not text.startswith(header + "\n"):
        return None
    body = text[len(header) + 1:]
    required = (
        "## Stock status semantics",
        "`status`: 1=available / 2=allocated / 3=in-transit",
        "Count only `status=1` as available",
        "Never sum all status values as available stock",
        "stock_balance_summary_v1",
    )
    if not all(part in body for part in required):
        return None
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _parse_beijing_time(value: Any) -> datetime | None:
    """Parse the contract's explicitly Beijing-local request timestamp."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone(timedelta(hours=8)))
    return parsed.astimezone(timezone.utc)


def _canonical_customer_source(customer: Mapping[str, Any]) -> str | None:
    identity = customer.get("erp_identity")
    source = identity.get("source") if isinstance(identity, Mapping) else None
    if isinstance(source, Mapping) and isinstance(source.get("system"), str):
        return source["system"]
    return None


def _sanitize_erp_row(tool: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Remove internal customer and ERP primary keys before model exposure."""
    private = {"customer_id", "customer_no", "customer_dept", "goods_id", "detail_id",
               "bill_id", "goods_detail_id", "sale_bill_id", "source_ref"}
    return {key: value for key, value in row.items() if key not in private}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BusinessTools:
    """Bounded read-only HTTP and MCP adapter used by the agent."""

    def __init__(self, settings: Any):
        self.settings = settings
        self.service_registry, self.mcp_registry, self.private = _read_settings(settings)
        self._products: dict[tuple[str, str], dict[str, Any]] = {}
        self._mcp_sessions: dict[str, str] = {}
        self._mcp_initialized: set[str] = set()
        self._mcp_advertised: dict[str, set[str]] = {}
        self._mcp_locks: dict[str, asyncio.Lock] = {}
        self._inventory_rule_verified = False
        self._inventory_rule_ref: str | None = None
        self._client = build_opener(ProxyHandler({}), _NoRedirect())

    def available_tools(self) -> list[dict[str, Any]]:
        return [{"name": item["name"], "description": item["description"],
                 "parameters": json.loads(json.dumps(item["parameters"]))}
                for item in TOOL_DEFINITIONS]

    @staticmethod
    def _scope(scope: Mapping[str, Any]) -> dict[str, str] | None:
        if not isinstance(scope, Mapping):
            return None
        tenant = scope.get("tenant_id")
        customer = scope.get("customer_ref")
        region = str(scope.get("sales_region") or "").upper()
        if not all(isinstance(v, str) and 1 <= len(v) <= 240 for v in (tenant, customer)) or region not in REGIONS:
            return None
        return {"tenant_id": tenant, "customer_ref": customer, "sales_region": region}

    @staticmethod
    def _region_scope(scope: Mapping[str, Any]) -> dict[str, str] | None:
        if not isinstance(scope, Mapping):
            return None
        tenant = scope.get("tenant_id")
        region = str(scope.get("sales_region") or "").upper()
        if not isinstance(tenant, str) or not tenant or region not in REGIONS:
            return None
        return {"tenant_id": tenant, "sales_region": region}

    def _service(self, name: str) -> tuple[str, str]:
        if name not in HTTP_SERVICES:
            raise IntegrationError("service_not_allowlisted")
        raw = self.service_registry.get("services", self.service_registry).get(name)
        if not isinstance(raw, Mapping):
            raise IntegrationError("service_not_configured")
        base = raw.get("base_url")
        token = raw.get("token")
        if not isinstance(base, str) or not isinstance(token, str) or len(token) < 16:
            raise IntegrationError("service_not_configured")
        parts = urlsplit(base)
        try:
            loopback = parts.hostname is not None and ipaddress.ip_address(parts.hostname).is_loopback
        except ValueError:
            loopback = parts.hostname in {"localhost"}
        if (parts.scheme != "https" and not (parts.scheme == "http" and loopback)) or parts.username or parts.password or parts.query or parts.fragment:
            raise IntegrationError("service_endpoint_invalid")
        return base.rstrip("/"), token

    def _http(self, service: str, method: str, path: str, *, params: dict | None = None,
              body: dict | None = None, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
        started = time.monotonic()
        call_id = str(uuid.uuid4())
        base, token = self._service(service)
        if not path.startswith("/v1/") or ".." in path.split("/") or "\\" in path:
            raise IntegrationError("service_path_invalid")
        url = base + path
        if params:
            url += "?" + urlencode(params, doseq=True)
        headers = {"Authorization": "Bearer " + token, "Accept": "application/json",
                   "X-Contract-Version": "1.0"}
        data = None
        if body is not None:
            data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        req = Request(url, data=data, headers=headers, method=method)
        try:
            with self._client.open(req, timeout=min(max(timeout, .1), 40)) as response:
                raw = response.read(MAX_BODY + 1)
                if len(raw) > MAX_BODY:
                    _trace_event({"call_id": call_id, "transport": "http", "service": service,
                        "method": method, "path": path, "params": params, "request": body,
                        "http_status": response.status, "status": "response_too_large",
                        "observed_at": _now(), "elapsed_ms": int((time.monotonic()-started)*1000)})
                    raise IntegrationError("service_response_too_large")
                result = json.loads(raw)
        except HTTPError as exc:
            _trace_event({"call_id": call_id, "transport": "http", "service": service,
                "method": method, "path": path, "params": params, "request": body,
                "http_status": exc.code, "status": "http_error", "error": f"http_{exc.code}",
                "observed_at": _now(), "elapsed_ms": int((time.monotonic()-started)*1000)})
            exc.close()
            raise IntegrationError(f"service_http_{exc.code}") from None
        except (URLError, TimeoutError, socket.timeout, OSError) as exc:
            _trace_event({"call_id": call_id, "transport": "http", "service": service,
                "method": method, "path": path, "params": params, "request": body,
                "status": "unavailable", "error": type(exc).__name__,
                "observed_at": _now(), "elapsed_ms": int((time.monotonic()-started)*1000)})
            raise IntegrationError("service_unavailable") from None
        except (UnicodeError, json.JSONDecodeError) as exc:
            _trace_event({"call_id": call_id, "transport": "http", "service": service,
                "method": method, "path": path, "params": params, "request": body,
                "status": "invalid_response", "error": type(exc).__name__,
                "observed_at": _now(), "elapsed_ms": int((time.monotonic()-started)*1000)})
            raise IntegrationError("service_response_invalid") from None
        if not isinstance(result, dict):
            _trace_event({"call_id": call_id, "transport": "http", "service": service,
                "method": method, "path": path, "params": params, "request": body,
                "http_status": response.status, "status": "invalid_response",
                "observed_at": _now(), "elapsed_ms": int((time.monotonic()-started)*1000)})
            raise IntegrationError("service_response_invalid")
        _trace_event({"call_id": call_id, "transport": "http", "service": service,
            "method": method, "path": path, "params": params, "request": body,
            "http_status": response.status, "status": "ok", "result": result,
            "observed_at": _now(), "elapsed_ms": int((time.monotonic()-started)*1000)})
        return result

    def _mcp_server(self) -> tuple[str, Mapping[str, Any]]:
        servers = self.mcp_registry.get("servers", {})
        if not isinstance(servers, Mapping):
            raise IntegrationError("mcp_not_configured")
        tool_refs = {str(t.get("server_ref")) for t in self.mcp_registry.get("brain_tools", [])
                     if isinstance(t, Mapping) and str(t.get("name", "")).startswith("erp_")
                     and t.get("enabled", True) is True}
        preferred = next(iter(tool_refs)) if len(tool_refs) == 1 else None
        for ref, server in servers.items():
            if preferred and str(ref) != preferred:
                continue
            if isinstance(server, Mapping) and server.get("enabled") is True:
                url = server.get("url")
                if not isinstance(url, str):
                    continue
                parts = urlsplit(url)
                try:
                    local = parts.hostname is not None and ipaddress.ip_address(parts.hostname).is_loopback
                except ValueError:
                    local = parts.hostname == "localhost"
                if parts.scheme != "http" or not local or parts.username or parts.password or parts.query or parts.fragment:
                    continue
                return str(ref), server
        raise IntegrationError("mcp_not_configured")

    def _mcp_source_system(self) -> str:
        try:
            ref, server = self._mcp_server()
        except IntegrationError:
            return "configured_external_mcp"
        return ("configured_local_mcp" if server.get("url", "").startswith("http://127.0.0.1:")
                else "configured_external_mcp")

    def _mcp_rpc_sync(self, url: str, server: Mapping[str, Any], method: str,
                      params: dict | None, timeout: float, *, notification: bool = False) -> dict[str, Any] | None:
        body: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if not notification:
            body["id"] = str(uuid.uuid4())
        if params is not None:
            body["params"] = params
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                   "MCP-Protocol-Version": str(server.get("protocol_version", "2024-11-05"))}
        session = self._mcp_sessions.get(url)
        if session:
            headers["Mcp-Session-Id"] = session
        req = Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with self._client.open(req, timeout=min(max(timeout, .1), 40)) as response:
                session_id = response.headers.get("Mcp-Session-Id")
                if session_id:
                    self._mcp_sessions[url] = session_id
                if notification:
                    return None
                raw = response.read(MAX_BODY + 1)
        except HTTPError as exc:
            status_code = exc.code
            exc.close()
            raise IntegrationError(f"mcp_http_{status_code}") from None
        except (URLError, TimeoutError, socket.timeout, OSError):
            raise IntegrationError("mcp_unavailable") from None
        if len(raw) > MAX_BODY:
            raise IntegrationError("mcp_response_too_large")
        text = raw.decode("utf-8", "strict")
        if "text/event-stream" in response.headers.get("Content-Type", ""):
            chunks = [line[6:].strip() for line in text.splitlines() if line.startswith("data: ")]
            text = next((line for line in reversed(chunks) if line), "")
        try:
            message = json.loads(text)
        except (ValueError, UnicodeError):
            raise IntegrationError("mcp_response_invalid") from None
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or "error" in message:
            raise IntegrationError("mcp_rpc_error")
        result = message.get("result")
        if not isinstance(result, dict):
            raise IntegrationError("mcp_result_invalid")
        return result

    async def _mcp_call(self, tool: str, args: dict, scope: dict, timeout: float = 25) -> dict[str, Any]:
        server_ref, server = self._mcp_server()
        allowed = server.get("allowed_tools")
        if (not isinstance(allowed, list) or tool not in allowed
                or tool not in READONLY_MCP_TOOLS):
            raise IntegrationError("mcp_tool_not_allowlisted")
        grant = server.get("scopes", {})
        if (scope["tenant_id"] not in grant.get("tenants", [])
                or scope.get("region", scope.get("sales_region")) not in grant.get("regions", [])
                or "read" not in grant.get("permissions", [])):
            raise IntegrationError("mcp_scope_forbidden")
        url = str(server["url"])
        deadline = time.monotonic() + timeout
        lock = self._mcp_locks.setdefault(url, asyncio.Lock())
        async def call(method: str, params: dict | None, notify: bool = False):
            left = max(.1, deadline - time.monotonic())
            started = time.monotonic()
            call_id = str(uuid.uuid4())
            try:
                result = await asyncio.to_thread(self._mcp_rpc_sync, url, server, method, params, left,
                                                 notification=notify)
            except IntegrationError as exc:
                code = str(exc)
                _trace_event({"call_id": call_id, "transport": "mcp", "server_ref": server_ref,
                    "tool": params.get("name") if isinstance(params, dict) else None,
                    "method": method, "request": params, "status": "error", "error": code,
                    "observed_at": _now(), "elapsed_ms": int((time.monotonic()-started)*1000)})
                raise
            _trace_event({"call_id": call_id, "transport": "mcp", "server_ref": server_ref,
                "tool": params.get("name") if isinstance(params, dict) else None,
                "method": method, "request": params, "status": "notification" if notify else "ok",
                "result": result, "observed_at": _now(),
                "elapsed_ms": int((time.monotonic()-started)*1000)})
            return result
        async with lock:
            if url not in self._mcp_initialized:
                await call("initialize", {"protocolVersion": str(server.get("protocol_version", "2024-11-05")),
                                           "capabilities": {}, "clientInfo": {"name": "sales-brain-pydantic-public", "version": "1.0"}})
                await call("notifications/initialized", {}, True)
                self._mcp_initialized.add(url)
                listed = await call("tools/list", {})
                names = {x.get("name") for x in listed.get("tools", []) if isinstance(x, dict)}
                self._mcp_advertised[url] = names & READONLY_MCP_TOOLS
            if tool not in self._mcp_advertised.get(url, set()):
                raise IntegrationError("mcp_tool_not_advertised")
            result = await call("tools/call", {"name": tool, "arguments": args})
        assert isinstance(result, dict)
        if result.get("isError") is True:
            raise IntegrationError("mcp_tool_error")
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return _unwrap_mcp_result(structured)
        blocks = result.get("content", [])
        for block in blocks:
            if isinstance(block, Mapping) and isinstance(block.get("text"), str):
                try:
                    parsed = json.loads(block["text"])
                except ValueError:
                    continue
                if isinstance(parsed, dict):
                    return _unwrap_mcp_result(parsed)
        if tool in {"mysql_query", "read_dictionary_doc"} and isinstance(blocks, list):
            return {"content": blocks}
        raise IntegrationError("mcp_result_invalid")

    async def _ctx(self, scope: Mapping[str, Any]) -> dict[str, Any]:
        s = self._scope(scope)
        if s is None:
            raise IntegrationError("trusted_scope_invalid")
        ref = quote(s["customer_ref"], safe="")
        value = await asyncio.to_thread(self._http, "customer-context", "GET",
            f"/v1/customers/{ref}/context", params={"aggregate": "false"}, timeout=8)
        customer = value.get("customer")
        if not isinstance(customer, dict) or customer.get("tenant_id") != s["tenant_id"] or customer.get("customer_ref") != s["customer_ref"] or customer.get("region") != s["sales_region"]:
            raise IntegrationError("customer_scope_mismatch")
        identity = customer.get("erp_identity") if isinstance(customer.get("erp_identity"), dict) else {}
        if scope.get("customer_id") is not None and str(identity.get("erp_customer_id", "")) != str(scope["customer_id"]):
            raise IntegrationError("erp_customer_id_scope_mismatch")
        if scope.get("customer_code") is not None and str(identity.get("customer_code", "")) != str(scope["customer_code"]):
            raise IntegrationError("erp_customer_code_scope_mismatch")
        return value

    async def resolve_scope(self, customer_id: str | int | None, sales_region: str | None,
                            tenant_id: str | None = None) -> dict[str, Any]:
        """Resolve numeric/external ERP customer ID only through exact profile identity."""
        tenant = tenant_id or _get(self.settings, "tenant_id", default="example-tenant") or "example-tenant"
        if customer_id is None or str(customer_id).strip() == "":
            return {"tenant_id": tenant, "customer_id": None, "customer_ref": None,
                    "sales_region": str(sales_region or "").upper(), "identity_status": "missing"}
        raw_id = str(customer_id).strip()
        if len(raw_id) > 160 or not raw_id.isprintable():
            return {"tenant_id": tenant, "customer_id": raw_id[:160], "customer_ref": None,
                    "sales_region": str(sales_region or "").upper(), "identity_status": "invalid"}
        regions = [str(sales_region).upper()] if sales_region else list(REGIONS)
        if any(region not in REGIONS for region in regions):
            return {"tenant_id": tenant, "customer_id": raw_id, "customer_ref": None,
                    "sales_region": str(sales_region or "").upper(), "identity_status": "invalid_region"}
        matches: list[dict[str, Any]] = []
        for region in regions:
            refs: list[str] = []
            cursor = None
            for _ in range(20):
                params: dict[str, Any] = {"query": raw_id, "region": region, "limit": 100}
                if cursor:
                    params["cursor"] = cursor
                page = await asyncio.to_thread(self._http, "customer-context", "GET", "/v1/customers", params=params, timeout=8)
                for row in page.get("items", []):
                    candidate_identity = row.get("erp_identity", {}) if isinstance(row, dict) else {}
                    exact = raw_id in {str(candidate_identity.get("customer_code", "")),
                                       str(candidate_identity.get("erp_customer_id", ""))}
                    pk = str(candidate_identity.get("erp_customer_id", ""))
                    canonical_ref = f"erp:{region}:{pk}" if pk else ""
                    if (exact and pk and row.get("customer_ref") == canonical_ref
                            and candidate_identity.get("region") == region):
                        refs.append(canonical_ref)
                cursor = page.get("next_cursor")
                if not cursor:
                    break
            if cursor:
                return {"tenant_id": tenant, "input_customer_id": raw_id, "customer_id": None,
                        "customer_code": None, "customer_ref": None,
                        "sales_region": region, "identity_status": "search_incomplete"}
            for ref in dict.fromkeys(r for r in refs if isinstance(r, str) and r):
                try:
                    candidate_scope = {"tenant_id": tenant,
                                       "customer_ref": ref, "sales_region": region}
                    ctx = await self._ctx(candidate_scope)
                except IntegrationError:
                    continue
                customer = ctx["customer"]
                identity = customer.get("erp_identity") if isinstance(customer.get("erp_identity"), dict) else {}
                execution = customer.get("execution_context") if isinstance(customer.get("execution_context"), dict) else {}
                source = _canonical_customer_source(customer)
                if (identity.get("status") == "confirmed"
                        and raw_id in {str(identity.get("erp_customer_id", "")),
                                       str(identity.get("customer_code", ""))}
                        and identity.get("region") == region
                        and str(customer.get("customer_ref")) == f"erp:{region}:{identity.get('erp_customer_id')}"
                        and customer.get("active") is True
                        and execution.get("simulation") is False
                        and execution.get("namespace") == "verified:" + str(customer.get("customer_ref"))
                        and source == "example_analytics.customer_profile"):
                    matches.append({"tenant_id": tenant, "customer_id": raw_id,
                                    "input_customer_id": raw_id,
                                    "customer_ref": ref, "sales_region": region,
                                    "erp_identity": identity})
        if len(matches) == 1:
            identity = matches[0]["erp_identity"]
            return {**matches[0], "customer_id": str(identity["erp_customer_id"]),
                    "customer_code": identity["customer_code"], "identity_status": "confirmed"}
        return {"tenant_id": tenant, "input_customer_id": raw_id, "customer_id": None,
                "customer_code": None, "customer_ref": None,
                "sales_region": str(sales_region or "").upper(),
                "identity_status": "ambiguous" if matches else "unresolved",
                "candidate_regions": sorted({m["sales_region"] for m in matches})}

    def _result(self, status: str, data: dict | None = None, evidence: list[dict] | None = None,
                missing: list[str] | None = None) -> dict[str, Any]:
        return {"status": status, "data": data or {}, "evidence": evidence or [],
                "missing_information": missing or []}

    def _evidence(self, source: str, content: Any, *, verified: bool, as_of: str | None = None,
                  observed_at: str | None = None, source_ref: str | None = None,
                  version: str | int | None = None, ref: str | None = None,
                  status: str = "current") -> dict[str, Any]:
        return {"evidence_ref": ref or "ev:" + _hash([source, content, as_of])[:32],
                "source": source, "source_ref": source_ref, "version": version,
                "as_of": as_of, "observed_at": observed_at, "status": status,
                "verified": verified, "content": content}

    def _remember_products(self, scope: Mapping[str, Any], rows: list[dict[str, Any]]) -> None:
        s = self._scope(scope)
        if not s:
            return
        run_id = str(scope.get("run_id") or "")
        for row in rows:
            ref = row.get("product_id") or row.get("product_ref") or row.get("stable_id")
            if isinstance(ref, str) and ref:
                self._products[(run_id, ref)] = {**row, "_scope": s}

    def _known_product(self, scope: Mapping[str, Any], ref: str) -> dict[str, Any] | None:
        s = self._scope(scope)
        if not s:
            return None
        row = self._products.get((str(scope.get("run_id") or ""), ref))
        if not isinstance(row, dict) or row.get("_scope") != s:
            return None
        return row

    async def execute(self, name: str, args: dict, scope: dict) -> dict:
        trace: list[dict[str, Any]] = []
        token = _TRANSPORT_TRACE.set(trace)
        try:
            result = await self._execute_inner(name, args, scope)
        except BaseException as exc:
            try:
                setattr(exc, "transport_trace", list(trace))
            except Exception:
                pass
            raise
        finally:
            _TRANSPORT_TRACE.reset(token)
        if isinstance(result, dict):
            result["transport_trace"] = trace
        return result

    async def _execute_inner(self, name: str, args: dict, scope: dict) -> dict:
        """Execute one fixed, read-only tool under caller-authenticated scope."""
        entry = next((x for x in TOOL_DEFINITIONS if x["name"] == name), None)
        if entry is None:
            return self._result("unavailable", missing=["tool_not_available"])
        try:
            _validate(entry["parameters"], args)
        except (TypeError, ValueError):
            return self._result("error", missing=["invalid_tool_arguments"])
        public_only = name in {"search_company_knowledge", "query_knowledge"}
        s = self._scope(scope)
        rs = self._region_scope(scope)
        expected_tenant = (self.private.get("SA_TENANT_ID")
                           or _get(self.settings, "tenant_id", "SA_TENANT_ID", default="example-tenant")
                           or "example-tenant")
        query_only = name == "query_knowledge"
        public_tenant = scope.get("tenant_id") if isinstance(scope, Mapping) else None
        if ((s and s["tenant_id"] != expected_tenant) or (rs and rs["tenant_id"] != expected_tenant)
                or (query_only and public_tenant != expected_tenant)):
            return self._result("unavailable", missing=["tenant_scope_mismatch"])
        has_tenant = isinstance(scope, Mapping) and isinstance(scope.get("tenant_id"), str) and bool(scope.get("tenant_id"))
        if ((name == "search_company_knowledge" and rs is None)
                or (query_only and not has_tenant)
                or (not public_only and s is None)):
            return self._result("unavailable", missing=["trusted_region_scope" if public_only else "trusted_customer_region_scope"])
        try:
            if name == "read_customer_context":
                return await self._read_context(scope)
            if name == "resolve_product_reference":
                return await self._resolve_product(args, scope)
            if name == "search_products":
                return await self._search_products(args, scope)
            if name == "erp_inventory_snapshot":
                return await self._inventory(args, scope)
            if name == "get_product_media":
                return await self._media(args, scope)
            if name == "resolve_quote":
                return await self._quote(args, scope)
            if name == "search_company_knowledge":
                return await self._company_knowledge(args, scope)
            if name == "query_knowledge":
                if self.private.get("KNOWLEDGE_READ_ONLY_CONFIRMED") is not True:
                    return self._result("unavailable", missing=["knowledge_profile_readonly_not_verified"])
                return await self._query_knowledge(args, scope)
            if name in {"erp_sample_card_lookup", "erp_order_progress", "erp_customer_debt",
                        "erp_product_facts", "erp_customer_activity"}:
                return await self._erp_tool(name, args, scope)
            if name == "read_workflow_context":
                return await self._workflow(args, scope)
            if name == "read_sales_capabilities":
                return await self._capabilities(args, scope)
            if name == "read_sales_owner":
                return await self._owner(args, scope)
        except IntegrationError as exc:
            code = str(exc)
            return self._result("unavailable", missing=[code])
        except (ValueError, TypeError, KeyError):
            return self._result("error", missing=["integration_contract_invalid"])
        return self._result("unavailable", missing=["tool_not_implemented"])

    async def _read_context(self, scope: dict) -> dict:
        raw = await self._ctx(scope)
        customer = raw["customer"]
        fence = _parse_beijing_time(scope.get("conversation_time_beijing"))
        rows = []
        for row in raw.get("messages", []):
            if not isinstance(row, dict):
                continue
            if row.get("customer_ref") not in (None, scope.get("customer_ref")) or row.get("role") not in {"customer", "sales"}:
                continue
            created = _parse_time(row.get("created_at"))
            if fence is None or created is None or created > fence:
                continue
            rows.append({k: row.get(k) for k in ("event_ref", "role", "text", "created_at", "source_revision") if k in row})
        identity = customer.get("erp_identity") if isinstance(customer.get("erp_identity"), dict) else {}
        source_status = {}
        for key, value in (raw.get("sources", {}) if isinstance(raw.get("sources"), dict) else {}).items():
            if isinstance(value, dict):
                source_status[str(key)] = {k: value.get(k) for k in ("status", "partial") if k in value}
        data = {"display_name": customer.get("name"),
                "region": customer.get("region"), "language": customer.get("language"),
                "erp_identity_confirmed": identity.get("status") == "confirmed" and identity.get("region") == customer.get("region"),
                "customer_grade": identity.get("customer_grade"), "customer_type": identity.get("customer_type"),
                "messages": rows[-60:], "source_status": source_status,
                "history_time_fence": scope.get("conversation_time_beijing")}
        return self._result("ok", data, [self._evidence("customer-context", data, verified=True,
            observed_at=raw.get("observed_at"))])

    async def _resolve_product(self, args: dict, scope: dict) -> dict:
        s = self._scope(scope)
        result = await asyncio.to_thread(self._http, "product-catalog", "GET", "/v1/products",
            params={"region": s["sales_region"], "code": args["reference"], "limit": 10}, timeout=12)
        rows = [r for r in result.get("items", []) if isinstance(r, dict)
                and r.get("region", s["sales_region"]) == s["sales_region"]
                and r.get("active") is not False and r.get("public") is not False]
        self._remember_products(scope, rows)
        if not rows:
            return self._result("not_found", {"complete": result.get("complete", True)}, [], ["exact_product_not_found"])
        return self._result("ok", {"products": rows, "complete": result.get("complete", True),
                                    "generation": result.get("generation")},
                            [self._evidence("product-catalog", rows, verified=True,
                                            observed_at=result.get("observed_at"), version=result.get("generation"))])

    async def _search_products(self, args: dict, scope: dict) -> dict:
        s = self._scope(scope)
        body = {"query": args["query"], "region": s["sales_region"],
                "customer_ref": s["customer_ref"], "limit": args.get("limit", 10),
                "conditions": args.get("conditions", [])}
        result = await asyncio.to_thread(self._http, "product-matching", "POST", "/v1/search", body=body, timeout=35)
        rows = result.get("qualified", result.get("items", []))
        if not isinstance(rows, list):
            raise IntegrationError("product_search_contract_invalid")
        safe_rows = [r for r in rows if isinstance(r, dict)
                     and r.get("region", s["sales_region"]) == s["sales_region"]
                     and r.get("active") is not False and r.get("public") is not False]
        self._remember_products(scope, safe_rows)
        coverage = result.get("coverage") if isinstance(result.get("coverage"), dict) else {}
        status = "partial" if result.get("partial") or coverage.get("complete") is False else "ok" if safe_rows else "not_found"
        data = {"products": safe_rows, "pending": result.get("pending", []),
                "coverage": coverage, "generation": result.get("generation")}
        return self._result(status, data, [self._evidence("product-matching", data, verified=True,
            observed_at=result.get("observed_at"), version=result.get("generation"))],
            [] if safe_rows else ["qualified_product_not_found"])

    async def _hydrate_product(self, scope: dict, ref: str) -> dict[str, Any]:
        row = self._known_product(scope, ref)
        if not row:
            raise IntegrationError("product_must_be_resolved_in_this_run")
        s = self._scope(scope)
        value = await asyncio.to_thread(self._http, "product-catalog", "GET",
            f"/v1/products/{quote(ref, safe='')}", params={"region": s["sales_region"]}, timeout=12)
        if value.get("region", s["sales_region"]) != s["sales_region"] or value.get("active") is False or value.get("public") is False:
            raise IntegrationError("product_scope_mismatch")
        if str(value.get("product_id") or value.get("stable_id")) != ref:
            raise IntegrationError("product_identity_mismatch")
        return value

    async def _inventory(self, args: dict, scope: dict) -> dict:
        product = await self._hydrate_product(scope, args["product_ref"])
        ref = args["product_ref"]
        ctx = await self._ctx(scope)
        identity = ctx["customer"].get("erp_identity", {})
        if identity.get("status") != "confirmed" or not identity.get("erp_customer_id") or identity.get("region") != self._scope(scope)["sales_region"]:
            return self._result("unavailable", missing=["confirmed_erp_customer_identity"])
        mcp_scope = {"tenant_id": scope["tenant_id"], "tenant_ref": scope["tenant_id"],
                     "customer_ref": scope["customer_ref"], "region": self._scope(scope)["sales_region"],
                     "environment": str(scope.get("environment", "live"))}
        raw = await self._mcp_call("business_guide", {"question": f"查询{mcp_scope['region']}仓货号{product.get('code')}的库存记录，按仓库、仓库类型、颜色、单位和状态分列，不查询成本或客户明细。", "resolved_question": f"查询{mcp_scope['region']}仓货号{product.get('code')}的库存记录，按仓库、仓库类型、颜色、单位和状态分列，不查询成本或客户明细。", "domain_hint": "available_stock"}, mcp_scope)
        guide = raw
        guide_id = guide.get("guide_id")
        recommended = guide.get("recommended_tables", [])
        has_inventory_table = isinstance(recommended, list) and any(
            item in {"example_reporting.inventory_barcode_detail_dw", "inventory_barcode_detail_dw"}
            for item in recommended if isinstance(item, str))
        if (guide.get("contract_conflict") or guide.get("matched_intent") != "available_stock"
                or guide.get("unsupported") or guide.get("clarify_required")
                or not isinstance(guide_id, str) or not has_inventory_table):
            return self._result("unavailable", {"stage": "business_guide"}, [], ["inventory_guide_not_verified"])
        code = product.get("code")
        goods_id = product.get("goods_id")
        if not isinstance(code, str) or type(goods_id) is not int or goods_id <= 0:
            return self._result("unavailable", missing=["catalog_goods_identity"])
        color_codes = args.get("color_codes", [])
        if any(not isinstance(x, str) or not x for x in color_codes):
            return self._result("error", missing=["invalid_color_codes"])
        color_sources: dict[str, list[str]] = {}
        if color_codes:
            indexed: dict[str, dict[str, Any]] = {}
            for message in ctx.get("messages", []):
                if isinstance(message, dict) and isinstance(message.get("event_ref"), str):
                    indexed[message["event_ref"]] = message
            scope_messages = scope.get("source_messages", [])
            if isinstance(scope_messages, list):
                for message in scope_messages:
                    if not isinstance(message, dict):
                        continue
                    event_ref = message.get("event_ref")
                    if (not isinstance(event_ref, str) or not event_ref
                            or message.get("role") not in {"customer", "sales"}
                            or not isinstance(message.get("text"), str)
                            or (message.get("customer_ref") is not None and message.get("customer_ref") != scope.get("customer_ref"))):
                        continue
                    old = indexed.get(event_ref)
                    if old is not None and (old.get("text") != message.get("text") or old.get("role") != message.get("role")):
                        return self._result("unavailable", missing=["ambiguous_color_source_event"])
                    indexed[event_ref] = message
            fence = _parse_beijing_time(scope.get("conversation_time_beijing"))
            if fence is None:
                return self._result("unavailable", missing=["conversation_time_for_color_source_fence"])
            requested_refs = args.get("source_event_refs", [])
            if requested_refs and any(ref not in indexed for ref in requested_refs):
                return self._result("unavailable", missing=["color_source_reference_not_in_customer_context"])
            eligible_refs = [ref for ref, message in indexed.items()
                if message.get("role") in {"customer", "sales"}
                and message.get("removed") not in (True, "true")
                and message.get("body_truncated") not in (True, "true")
                and isinstance(message.get("text"), str)
                and (created := _parse_beijing_time(message.get("created_at"))) is not None and created <= fence]
            search_refs = requested_refs or eligible_refs
            for color in color_codes:
                bound = []
                for source_ref in search_refs:
                    message = indexed.get(source_ref)
                    created = _parse_beijing_time(message.get("created_at")) if isinstance(message, dict) else None
                    if (isinstance(message, dict) and message.get("role") in {"customer", "sales"}
                            and isinstance(message.get("text"), str) and color in message["text"]
                            and message.get("removed") not in (True, "true")
                            and message.get("body_truncated") not in (True, "true")
                            and created is not None and created <= fence):
                        bound.append(source_ref)
                if not bound:
                    return self._result("unavailable", missing=["color_source_not_verified"])
                color_sources[color] = bound
        safe_colors = [_sql_literal(x) for x in color_codes]
        color_clause = " AND attr_val IN (" + ",".join("'" + x + "'" for x in safe_colors) + ")" if safe_colors else ""
        fields = "goods_id, goods_no, whse_dept, whse_name, whse_type, attr_val, unit, status, is_missing, is_ccbs_whse, is_defective_whse"
        sql = (f"SELECT {fields}, COUNT(*) AS record_count, SUM(piece_num) AS pieces, SUM(goods_num) AS qty "
               "FROM example_reporting.inventory_barcode_detail_dw WHERE "
               f"goods_id={goods_id} AND goods_no='{code.replace(chr(39), chr(39)*2)}' "
               f"AND whse_dept='{self._scope(scope)['sales_region']}' "
               "AND (sale_bill_id IS NULL OR (is_delete='n' AND is_void='n')) "
               f"{color_clause} GROUP BY {fields} ORDER BY whse_name, attr_val, unit, status LIMIT 50")
        valid = await self._mcp_call("validate_readonly_sql", {"sql": sql, "active_database": "example_reporting", "guide_context_id": guide_id}, mcp_scope)
        if valid.get("allowed") is not True or valid.get("business_errors") != [] or any(valid.get(k) != [] for k in ("high_risk_findings", "business_warnings", "warnings")):
            return self._result("unavailable", {"stage": "validate_readonly_sql"}, [], ["inventory_query_validation_not_passed"])
        question = f"查询货号{code}在{self._scope(scope)['sales_region']}地区按仓库、颜色、单位和库存状态汇总的记录"
        result = await self._mcp_call("mysql_query", {"sql": sql, "active_database": "example_reporting", "guide_context_id": guide_id, "question": question, "domain_hint": "available_stock"}, mcp_scope)
        rows, truncated = _extract_rows(result)
        safe = []
        for row in rows:
            if str(row.get("goods_id")) != str(goods_id) or row.get("goods_no") != code or row.get("whse_dept") != self._scope(scope)["sales_region"]:
                raise IntegrationError("inventory_row_scope_mismatch")
            safe.append({k: row.get(k) for k in ("goods_no", "whse_dept", "whse_name", "whse_type", "attr_val", "unit", "status", "is_missing", "is_ccbs_whse", "is_defective_whse", "record_count", "pieces", "qty")})
        origin = self._mcp_source_system()
        if not self._inventory_rule_verified:
            rule = await self._mcp_call("read_dictionary_doc",
                {"path": "example_analytics/stock-query-guide.md"}, mcp_scope)
            doc_text = "\n".join(b.get("text", "") for b in rule.get("content", [])
                                  if isinstance(b, dict) and isinstance(b.get("text"), str))
            doc_hash = _inventory_dictionary_digest(doc_text, origin)
            self._inventory_rule_verified = doc_hash is not None
            self._inventory_rule_ref = f"configured-mcp:stock-guide-v1:{doc_hash}:live" if doc_hash else None
        labels = {1: "未配", 2: "已配", 3: "在途"} if self._inventory_rule_verified else {}
        projected = []
        for row in safe:
            flags = ("is_missing", "is_ccbs_whse", "is_defective_whse")
            try:
                qty = Decimal(str(row.get("qty")))
                pieces = Decimal(str(row.get("pieces")))
                positive = qty.is_finite() and pieces.is_finite() and qty > 0 and pieces > 0
            except (InvalidOperation, ValueError, TypeError):
                positive = False
            warehouse_known = self._inventory_rule_verified and row.get("whse_type") in {"正品仓", "米样仓"}
            normal_bulk = (self._inventory_rule_verified and origin == "configured_local_mcp"
                           and row.get("whse_type") == "正品仓"
                           and all(row.get(key) == "n" for key in flags))
            available = normal_bulk and row.get("status") == 1 and positive
            projected.append({**row,
                "inventory_state": labels.get(row.get("status")),
                "unit_name": {"y": "yard", "m": "meter"}.get(row.get("unit")),
                "unit_label_zh": {"y": "码", "m": "米"}.get(row.get("unit")),
                "availability_rule_ref": self._inventory_rule_ref if origin == "configured_local_mcp" and self._inventory_rule_verified else None,
                "warehouse_use_verified": bool(warehouse_known),
                "normal_bulk_warehouse_verified": bool(normal_bulk),
                "available_stock_verified": bool(available),
                "availability_meaning": "positive_unallocated_normal_goods_stock" if available else "not_established_as_available_bulk_stock",
                "sale_eligibility_verified": False,
                "shipment_commitment_verified": False})
        limited = bool(truncated or len(projected) >= 50)
        recorded_colors = sorted({str(row.get("attr_val")) for row in projected if row.get("attr_val") is not None})
        unreturned_colors = sorted(set(color_codes) - set(recorded_colors))
        color_coverage_complete = bool(color_codes) and not unreturned_colors
        content = {"product_ref": ref, "records": projected, "complete": not limited,
                   "read_at": _now(), "availability_rule_verified": self._inventory_rule_verified,
                   "requested_color_codes": color_codes, "recorded_color_codes": recorded_colors,
                   "unreturned_color_codes_not_zero": unreturned_colors,
                   "color_coverage_complete": color_coverage_complete if color_codes else None,
                   "color_source_bindings": color_sources,
                   "meaning": "Observed warehouse records only. No matching rows do not establish zero stock; this does not reserve, approve sale, or promise shipment.",
                   "unit_note": "Keep source unit exactly; y means yards/码 where returned. pieces is the ERP piece count and is not automatically a roll count."}
        status = "partial" if safe and (not self._inventory_rule_verified or limited or bool(unreturned_colors)) else "ok" if safe else "not_found"
        source_refs = result.get("source_refs", [])
        source_ref = source_refs[0] if isinstance(source_refs, list) and source_refs and isinstance(source_refs[0], str) else None
        evidence = [self._evidence("erp-inventory-mcp", row, verified=True,
                                   observed_at=content["read_at"], source_ref=source_ref,
                                   version=row.get("availability_rule_ref"),
                                   status="current_snapshot") for row in projected]
        missing = []
        if not safe:
            missing.append("no_inventory_records_not_zero")
        if not self._inventory_rule_verified:
            missing.append("live_inventory_dictionary_rule_not_verified")
        if unreturned_colors:
            missing.append("no_record_for_requested_color_not_zero")
        if truncated:
            missing.append("mcp_result_truncated")
        elif len(projected) >= 50:
            missing.append("inventory_query_limit_reached")
        return self._result(status, content, evidence, missing)

    async def _media(self, args: dict, scope: dict) -> dict:
        row = await self._hydrate_product(scope, args["product_ref"])
        result = await asyncio.to_thread(self._http, "product-catalog", "GET",
            f"/v1/products/{quote(args['product_ref'], safe='')}/media",
            params={"region": self._scope(scope)["sales_region"]}, timeout=12)
        items = result.get("items", [])
        return self._result("ok" if items else "not_found", {"product_ref": args["product_ref"], "items": items},
            [self._evidence("product-catalog-media", items, verified=True, observed_at=result.get("observed_at"))] if items else [],
            [] if items else ["product_media_unavailable"])

    async def _quote(self, args: dict, scope: dict) -> dict:
        products = [await self._hydrate_product(scope, ref) for ref in args["product_refs"]]
        s = self._scope(scope)
        agent = scope.get("agent_id")
        body = {"product_ids": [p.get("product_id") for p in products], "customer_ref": s["customer_ref"],
                "region": s["sales_region"], "quantity": args.get("quantity"), "unit": args.get("unit"),
                "currency": args.get("currency"), "agent_id": agent if isinstance(agent, str) else None}
        result = await asyncio.to_thread(self._http, "pricing", "POST", "/v1/quotes", body=body, timeout=18)
        items = result.get("items", result.get("quotes", []))
        evidence = [self._evidence("pricing", item, verified=item.get("valid") is True,
                                   as_of=item.get("as_of"), observed_at=item.get("observed_at") or result.get("observed_at"),
                                   source_ref=item.get("quote_ref"), version=item.get("price_policy_revision"))
                    for item in items if isinstance(item, dict)]
        status = "ok" if any(e["verified"] for e in evidence) else "partial" if items else "not_found"
        missing = [] if status == "ok" else ["no_verified_applicable_quote"]
        return self._result(status, {"items": items, "terms": {k: body[k] for k in ("quantity", "unit", "currency") if body.get(k) is not None}}, evidence, missing)

    async def _company_knowledge(self, args: dict, scope: dict) -> dict:
        s = self._region_scope(scope)
        result = await asyncio.to_thread(self._http, "company-knowledge", "POST", "/v1/search",
            body={"query": args["question"], "region": s["sales_region"],
                  "brand": scope.get("brand"), "limit": 10}, timeout=12)
        items = result.get("items", [])
        now = datetime.now(timezone.utc)
        current_items = [item for item in items if isinstance(item, dict) and item.get("public") is True
                         and item.get("source_ref") and item.get("valid_until")
                         and (validity := _parse_time(item.get("valid_until"))) is not None and validity > now
                         and item.get("region") in (s["sales_region"], "global")]
        evidence = [self._evidence("company-knowledge", item, verified=True,
                                   as_of=item.get("verified_at"), observed_at=result.get("observed_at"),
                                   source_ref=item.get("source_ref"), version=item.get("revision"),
                                   ref="ck:" + str(item.get("document_id")) + ":" + str(item.get("revision")),
                                   status="published_public_current")
                    for item in current_items]
        return self._result("ok" if evidence else "not_found", {"items": current_items}, evidence,
                            [] if evidence else ["no_published_company_reference"])

    async def _query_knowledge(self, args: dict, scope: dict) -> dict:
        base = self.private.get("KNOWLEDGE_BASE_URL") or _get(self.settings, "knowledge_base_url")
        token = self.private.get("KNOWLEDGE_API_TOKEN") or _get(self.settings, "knowledge_api_token")
        model = self.private.get("KNOWLEDGE_MODEL") or _get(self.settings, "knowledge_model", default="example-model")
        parts = urlsplit(base or "")
        endpoint = (parts.hostname, parts.port)
        if (not isinstance(base, str) or parts.scheme != "http"
                or endpoint not in {("192.0.2.10", 8650), ("127.0.0.1", 8650)}
                or parts.path.rstrip("/") != "/v1" or parts.username or parts.password
                or parts.query or parts.fragment or not isinstance(token, str) or not token):
            raise IntegrationError("knowledge_provider_not_configured")
        body = json.dumps({"model": model, "messages": [
            {"role": "system", "content": "仅根据你可访问的知识资料回答并尽可能返回出处。若没有依据请明确说不知道。你的回答是参考信息，不是已核验的当前价格、库存、订单或交付事实。"},
            {"role": "user", "content": args["question"]}], "temperature": 0}, ensure_ascii=False).encode()
        request_body = json.loads(body)
        started = time.monotonic()
        call_id = str(uuid.uuid4())
        request = Request(base.rstrip("/") + "/chat/completions", data=body,
                          headers={"Authorization": "Bearer " + token, "Content-Type": "application/json",
                                   "Accept": "application/json"}, method="POST")
        try:
            with self._client.open(request, timeout=20) as response:
                raw = response.read(256_001)
                if len(raw) > 256_000:
                    _trace_event({"call_id": call_id, "transport": "http", "service": "secondary-knowledge-model",
                        "method": "POST", "path": "/v1/chat/completions", "request": request_body,
                        "http_status": response.status, "status": "response_too_large",
                        "observed_at": _now(), "elapsed_ms": int((time.monotonic()-started)*1000)})
                    raise IntegrationError("knowledge_response_too_large")
                result = json.loads(raw)
        except HTTPError as exc:
            _trace_event({"call_id": call_id, "transport": "http", "service": "secondary-knowledge-model",
                "method": "POST", "path": "/v1/chat/completions", "request": request_body,
                "http_status": exc.code, "status": "http_error", "error": f"http_{exc.code}",
                "observed_at": _now(), "elapsed_ms": int((time.monotonic()-started)*1000)})
            exc.close()
            raise IntegrationError(f"knowledge_provider_http_{exc.code}") from None
        except (URLError, TimeoutError, socket.timeout, OSError, json.JSONDecodeError) as exc:
            _trace_event({"call_id": call_id, "transport": "http", "service": "secondary-knowledge-model",
                "method": "POST", "path": "/v1/chat/completions", "request": request_body,
                "status": "unavailable", "error": type(exc).__name__, "observed_at": _now(),
                "elapsed_ms": int((time.monotonic()-started)*1000)})
            raise IntegrationError("knowledge_provider_unavailable") from None
        choices = result.get("choices", []) if isinstance(result, dict) else []
        message = choices[0].get("message", {}) if choices else {}
        answer = message.get("content") if isinstance(message, dict) else None
        if not isinstance(answer, str):
            _trace_event({"call_id": call_id, "transport": "http", "service": "secondary-knowledge-model",
                "method": "POST", "path": "/v1/chat/completions", "request": request_body,
                "status": "invalid_response", "http_status": response.status,
                "result": result, "observed_at": _now(),
                "elapsed_ms": int((time.monotonic()-started)*1000)})
            raise IntegrationError("knowledge_provider_contract_invalid")
        content = {"answer": answer[:12000], "model": result.get("model", model),
                   "source_metadata": result.get("sources", result.get("citations", [])),
                   "verified_business_fact": False}
        evidence = self._evidence("secondary-knowledge-model", content, verified=False,
                                  as_of=None, observed_at=_now(), status="unverified_reference")
        _trace_event({"call_id": call_id, "transport": "http", "service": "secondary-knowledge-model",
            "method": "POST", "path": "/v1/chat/completions", "request": request_body,
            "status": "ok", "http_status": response.status, "result": result,
            "observed_at": _now(), "elapsed_ms": int((time.monotonic()-started)*1000)})
        return self._result("ok", content, [evidence], [])

    async def _erp_tool(self, name: str, args: dict, scope: dict) -> dict:
        # The current deployed MCP server exposes these six business tools. ERP
        # customer identity is read from the trusted customer profile, never args.
        ctx = await self._ctx(scope)
        identity = ctx["customer"].get("erp_identity")
        region = self._scope(scope)["sales_region"]
        if (not isinstance(identity, dict) or identity.get("status") != "confirmed"
                or identity.get("region") != region or not identity.get("erp_customer_id")
                or not identity.get("customer_code")):
            return self._result("unavailable", missing=["confirmed_erp_customer_identity"])
        args = dict(args)
        if name == "erp_order_progress" and not args.get("bill_no"):
            return self._result("unavailable", missing=["exact_order_number"])
        if name == "erp_sample_card_lookup" and not (args.get("goods_no") or args.get("sample_card_no")):
            return self._result("unavailable", missing=["product_or_sample_card_number"])
        scope_mcp = {"tenant_id": scope["tenant_id"], "tenant_ref": scope["tenant_id"],
                     "customer_ref": scope["customer_ref"], "region": region,
                     "environment": str(scope.get("environment", "live"))}
        try:
            sql, database, question, expected = self._erp_sql(name, args, scope, identity)
        except IntegrationError as exc:
            return self._result("unavailable", missing=[str(exc)])
        guide_args = {"question": question, "resolved_question": question}
        guide_hint = {"erp_sample_card_lookup": "sample_card", "erp_order_progress": "order_progress",
                      "erp_customer_debt": "receivable"}.get(name, expected)
        if guide_hint:
            guide_args["domain_hint"] = guide_hint
        guide_result = await self._mcp_call("business_guide", guide_args, scope_mcp)
        if guide_result.get("contract_conflict") or guide_result.get("unsupported") or guide_result.get("clarify_required"):
            return self._result("unavailable", {"stage": "business_guide"}, [], ["business_guide_requires_review"])
        guide_id = guide_result.get("guide_id")
        if not isinstance(guide_id, str) or not guide_id:
            return self._result("unavailable", {"stage": "business_guide"}, [], ["business_guide_invalid"])
        if name in {"erp_sample_card_lookup", "erp_order_progress", "erp_customer_debt"}:
            wanted = {"erp_sample_card_lookup": "sample_card", "erp_order_progress": "order_progress", "erp_customer_debt": "receivable"}[name]
            if guide_result.get("matched_intent") != wanted:
                return self._result("unavailable", {"stage": "business_guide"}, [], ["business_guide_intent_mismatch"])
        validated = await self._mcp_call("validate_readonly_sql", {
            "sql": sql, "active_database": database, "guide_context_id": guide_id}, scope_mcp)
        if (validated.get("allowed") is not True or validated.get("business_errors") != []
                or any(validated.get(k) != [] for k in ("high_risk_findings", "business_warnings", "warnings"))):
            return self._result("unavailable", {"stage": "validate_readonly_sql"}, [], ["readonly_sql_not_approved"])
        query_args = {"sql": sql, "active_database": database, "guide_context_id": guide_id,
                      "question": question}
        if expected:
            query_args["domain_hint"] = expected
        result = await self._mcp_call("mysql_query", query_args, scope_mcp, timeout=35)
        rows, truncated = _extract_rows(result)
        if result.get("status") not in (None, "ok", "success"):
            return self._result("unavailable", {"stage": "mysql_query"}, [], ["erp_query_failed"])
        for row in rows:
            if "customer_id" in row and str(row["customer_id"]) != str(identity["erp_customer_id"]):
                raise IntegrationError("erp_row_customer_mismatch")
            if "customer_no" in row and str(row["customer_no"]) != str(identity["customer_code"]):
                raise IntegrationError("erp_row_customer_code_mismatch")
            if "customer_dept" in row and row["customer_dept"] != region:
                raise IntegrationError("erp_row_region_mismatch")
        visible_rows = [_sanitize_erp_row(name, row) for row in rows]
        row_limit = 10 if name == "erp_product_facts" else int(args.get("limit", 20)) if name == "erp_customer_activity" else 20
        limited = bool(truncated or len(rows) >= row_limit)
        data = {"rows": visible_rows, "source": "configured-mcp",
                "meaning": {"erp_sample_card_lookup": "customer_sample_card_application_records_only",
                            "erp_order_progress": "customer_order_header_status_only",
                            "erp_customer_debt": "current_month_rmb_debt_snapshot",
                            "erp_customer_activity": "bounded_customer_owned_activity_records",
                            "erp_product_facts": "current_public_product_specs"}.get(name),
                "complete": not limited, "truncated": truncated, "read_at": _now(),
                "region": region}
        status = "partial" if rows and limited else "ok" if rows else "not_found"
        source_refs = result.get("source_refs", [])
        source_ref = source_refs[0] if isinstance(source_refs, list) and source_refs and isinstance(source_refs[0], str) else None
        evidence = [self._evidence("configured-mcp", row, verified=True, as_of=None,
                                   observed_at=data["read_at"], source_ref=source_ref,
                                   status="bounded_read") for row in visible_rows]
        missing = [] if rows and not limited else ["no_matching_erp_rows_not_zero"] if not rows else ["bounded_query_window"]
        if truncated:
            missing.append("mcp_result_truncated")
        return self._result(status, data, evidence, missing)

    def _erp_sql(self, name: str, args: dict, scope: dict, identity: dict) -> tuple[str, str, str, str]:
        """Reviewed fixed SELECT templates from the deployed MCP adapter.

        Model text is never inserted into SQL. Identifiers are literal-escaped,
        and customer identity/region come only from the confirmed profile.
        """
        cid = str(identity.get("erp_customer_id", ""))
        code = _sql_literal(str(identity.get("customer_code", "")))
        region = self._scope(scope)["sales_region"]
        if not cid.isdecimal() or not code:
            raise IntegrationError("confirmed_erp_customer_identity")
        if name == "erp_sample_card_lookup":
            goods = _sql_literal(args.get("goods_no", "")) if args.get("goods_no") else None
            sample = _sql_literal(args.get("sample_card_no", "")) if args.get("sample_card_no") else None
            filt = f"goods_no='{goods}'" if goods else f"bill_no='{sample}'" if sample else ""
            sql = ("SELECT customer_id, goods_no, apply_num, bill_no, detail_id "
                   "FROM example_analytics.sample_card_apply_detail_dwd "
                   f"WHERE customer_id={int(cid)} AND {filt} "
                   "ORDER BY bill_time DESC, detail_id DESC LIMIT 20")
            return sql, "example_analytics", "查询该客户对应货号或样卡申请编号的样卡申请明细", "sample_card"
        if name == "erp_order_progress":
            bill = _sql_literal(args.get("bill_no", "")) if args.get("bill_no") else None
            if not bill:
                raise IntegrationError("exact_order_number")
            sql = ("SELECT customer_id, bill_no, bill_id, bill_status, "
                   "DATE_FORMAT(sale_time, '%Y-%m-%d') AS bill_date "
                   "FROM example_transactions.sale_bill_ods "
                   f"WHERE customer_id={int(cid)} AND bill_no='{bill}' AND is_delete='n' AND is_void='n' "
                   "ORDER BY sale_time DESC LIMIT 20")
            return sql, "example_transactions", f"查询该客户订单{bill}的主单状态和订单日期", "order_progress"
        if name == "erp_customer_debt":
            sql = ("SELECT customer_id, customer_no, currency_no, debt_amount_rmb, bill_date, "
                   "DATE_FORMAT(CURDATE(), '%Y-%m') AS query_month "
                   "FROM example_reporting.customer_debt_bymonth_dw "
                   f"WHERE customer_id={int(cid)} AND customer_no='{code}' AND is_inner_cus='n' "
                   "AND bill_date=DATE_FORMAT(CURDATE(), '%Y-%m') LIMIT 20")
            return sql, "example_reporting", "查询该客户本月人民币应收账款快照", "receivable"
        if name == "erp_product_facts":
            refs = args.get("product_refs", [])
            products = [self._known_product(scope, ref) for ref in refs]
            if not products or any(p is None for p in products):
                raise IntegrationError("product_must_be_resolved_in_this_run")
            clauses, codes = [], []
            for p in products:
                gid = p.get("goods_id")
                code_value = _sql_literal(str(p.get("code", "")))
                if type(gid) is not int or gid <= 0 or not code_value:
                    raise IntegrationError("catalog_goods_identity")
                clauses.append(f"(goods_id={gid} AND goods_no='{code_value}')")
                codes.append(code_value)
            columns = "goods_id, goods_no, unit, category_name, component, min_weight, weight, width, m_kg, kg_m_num, technology, weaving_way, usage, pattern, moq_num, min_days, max_days"
            sql = (f"SELECT {columns} FROM example_analytics.goods_detail_dwd WHERE ({' OR '.join(clauses)}) "
                   "AND is_delete='n' AND is_void='n' ORDER BY goods_id LIMIT 10")
            return sql, "example_analytics", "查询已核验货号的公开面料规格", ""
        if name == "erp_customer_activity":
            topic = args["topic"]
            limit = int(args.get("limit", 20))
            date_filters = []
            for key, op in (("date_start", ">="), ("date_end", "<")):
                value = args.get(key)
                if value:
                    try:
                        datetime.strptime(value, "%Y-%m-%d")
                    except ValueError:
                        raise IntegrationError("business_date_invalid") from None
                    date_filters.append(f"{key}='{value}'")
            if len(date_filters) == 2 and args["date_start"] >= args["date_end"]:
                raise IntegrationError("business_date_range_invalid")
            if topic == "samples":
                filters = [f"customer_id={int(cid)}"]
                columns = "customer_id, bill_no, bill_time, goods_no, detail_id, apply_num, sample_card_type, bill_status, detail_status, send_time, estimate_receive_time"
                table, date_col = "example_analytics.sample_card_apply_detail_dwd", "bill_time"
                if args.get("goods_no"):
                    filters.append(f"goods_no='{_sql_literal(args['goods_no'])}'")
                if args.get("bill_no"):
                    filters.append(f"bill_no='{_sql_literal(args['bill_no'])}'")
                if args.get("color_no"):
                    raise IntegrationError("sample_color_filter_not_supported")
                hint = "sample_card"
            elif topic == "refunds":
                filters = [f"customer_no='{code}'"]
                columns = "customer_no, bill_no, bill_time, review_time, currency_no, bill_return_amount, bill_refundable_amount, bill_deal_amount, bill_status, bill_verification_status"
                table, date_col = "example_transactions.receive_return_bill_ods", "bill_time"
                if args.get("bill_no"):
                    filters.append(f"bill_no='{_sql_literal(args['bill_no'])}'")
                if args.get("goods_no") or args.get("color_no"):
                    raise IntegrationError("refund_product_link_unverified")
                hint = ""
            else:
                select = ["d.customer_id", "d.customer_no", "d.customer_dept", "d.bill_no", "d.goods_no",
                          "d.attr_val", "d.unit", "d.sale_goods_num", "d.sale_piece_num", "d.goods_detail_id",
                          "d.bill_status", "d.bill_type", "d.goods_detail_status", "d.sale_time",
                          "d.expected_arrival_time", "h.ship_time AS promised_delivery_date",
                          "h.delivery_way AS delivery_method", "h.delivery_fee_type",
                          "h.external_bill_no", "h.bill_type AS ods_bill_type"]
                filters = [f"d.customer_id={int(cid)}", f"d.customer_no='{code}'", f"d.customer_dept='{region}'", "d.is_inner_cus='n'", "h.is_delete='n'", "h.is_void='n'", f"h.customer_id={int(cid)}"]
                if args.get("goods_no"):
                    filters.append(f"d.goods_no='{_sql_literal(args['goods_no'])}'")
                if args.get("bill_no"):
                    filters.append(f"d.bill_no='{_sql_literal(args['bill_no'])}'")
                if args.get("color_no"):
                    filters.append(f"d.attr_val='{_sql_literal(args['color_no'])}'")
                if args.get("order_kind"):
                    dt, ot = {"meter_sample": ("SQ", 2), "bulk": ("bulk", 1)}[args["order_kind"]]
                    filters += [f"d.bill_type='{dt}'", f"h.bill_type={ot}"]
                if topic == "logistics":
                    filters.append("d.bill_status='6'")
                    select += ["d.delivery_time", "d.sorted_goods_num", "d.sorted_piece_num", "l.bill_no AS tracking_reference", "l.bill_time AS carrier_dispatch_record_at", "l.estimate_receive_time AS carrier_estimated_arrival_at"]
                    join = "JOIN example_transactions.sale_bill_ods h ON h.bill_id=d.sale_bill_id LEFT JOIN example_transactions.logistics_track_bill_ods l ON l.bill_id=h.track_bill_id AND l.is_void='n' AND l.is_delete='n'"
                else:
                    join = "JOIN example_transactions.sale_bill_ods h ON h.bill_id=d.sale_bill_id"
                table = "example_analytics.sale_bill_goods_detail_dwd d " + join
                columns = ", ".join(select)
                date_col = "d.sale_time"
                hint = "order_list"
            for key, op in (("date_start", ">="), ("date_end", "<")):
                if args.get(key):
                    filters.append(f"{date_col} {op} '{args[key]}'")
            if topic in {"samples", "refunds"}:
                sql = f"SELECT {columns} FROM {table} WHERE {' AND '.join(filters)} ORDER BY {date_col} DESC LIMIT {limit}"
                database = table.split(".", 1)[0]
            else:
                sql = f"SELECT {columns} FROM {table} WHERE {' AND '.join(filters)} ORDER BY {date_col} DESC LIMIT {limit}"
                database = "example_analytics"
            return sql, database, f"查询该客户{topic}记录", hint
        raise IntegrationError("erp_tool_not_supported")

    async def _workflow(self, _args: dict, scope: dict) -> dict:
        s = self._scope(scope)
        path = f"/v1/customers/{quote(s['customer_ref'], safe='')}/cases"
        result = await asyncio.to_thread(self._http, "sales-workflows", "GET", path,
                                         params={"region": s["sales_region"]}, timeout=10)
        safe = {k: result.get(k, []) for k in ("items", "requests")}
        ev = self._evidence("sales-workflows", safe, verified=True, as_of=result.get("observed_at"))
        return self._result("ok", safe, [ev])

    async def _capabilities(self, _args: dict, scope: dict) -> dict:
        s = self._scope(scope)
        path = f"/v1/customers/{quote(s['customer_ref'], safe='')}/capabilities"
        result = await asyncio.to_thread(self._http, "sales-workflows", "GET", path,
                                         params={"region": s["sales_region"]}, timeout=10)
        return self._result("ok", result, [self._evidence("sales-workflows-capabilities", result, verified=True)])

    async def _owner(self, _args: dict, scope: dict) -> dict:
        s = self._scope(scope)
        path = f"/v1/customers/{quote(s['customer_ref'], safe='')}/owner"
        result = await asyncio.to_thread(self._http, "sales-workflows", "GET", path,
                                         params={"region": s["sales_region"]}, timeout=10)
        owner = result.get("owner", result)
        if not isinstance(owner, dict) or not owner:
            return self._result("not_found", {}, [], ["sales_owner_unavailable"])
        visible = {k: owner[k] for k in ("name", "display_name", "public_name", "salesperson_name", "role", "team_name") if k in owner}
        if not visible:
            return self._result("unavailable", {}, [], ["public_sales_owner_name_unavailable"])
        return self._result("ok", visible, [self._evidence("sales-workflows-owner", visible, verified=True)])


def _extract_rows(result: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Decode the deployed MCP's rows or timing-prefixed JSON text envelope.

    Unknown/malformed responses raise instead of being mistaken for an empty
    business result. The server may return `rows`, `value`, `text`, or `texts`.
    """
    source: Any = result
    if isinstance(source, Mapping) and "data" in source and isinstance(source["data"], Mapping):
        source = source["data"]
    if not isinstance(source, Mapping):
        raise IntegrationError("mcp_query_shape_invalid")
    present = [k for k in ("rows", "value", "text", "texts") if k in source]
    if len(present) != 1:
        blocks = result.get("content", []) if isinstance(result.get("content"), list) else []
        texts = [b.get("text") for b in blocks if isinstance(b, Mapping) and isinstance(b.get("text"), str)]
        if not texts:
            raise IntegrationError("mcp_query_shape_invalid")
        value: Any = "\n\n".join(texts)
    else:
        value = source[present[0]]
        if present[0] == "texts":
            if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value):
                raise IntegrationError("mcp_query_text_invalid")
            value = "\n\n".join(value)
    if isinstance(value, list):
        if not all(isinstance(x, dict) for x in value):
            raise IntegrationError("mcp_query_rows_invalid")
        return value, False
    if isinstance(value, dict):
        return _extract_rows(value)
    if not isinstance(value, str) or len(value) > MAX_BODY:
        raise IntegrationError("mcp_query_text_invalid")
    text = value.strip()
    starts = list(re.finditer(r"(?:\A|\n\s*\n)([ \t]*\[)", text))
    parsed: list[tuple[list, bool]] = []
    for candidate in starts:
        start = candidate.start(1)
        leading = len(text[start:]) - len(text[start:].lstrip())
        try:
            rows, end = json.JSONDecoder().raw_decode(text[start + leading:])
        except ValueError:
            continue
        tail = text[start + leading + end:].strip()
        execution_timing = re.search(r"(?:\A|\n)Query execution time: ([0-9]+(?:\.[0-9]+)?) ms\Z", tail)
        if execution_timing:
            tail = tail[:execution_timing.start()].strip()
        cap = re.fullmatch(r"\.\.\.\(结果共 ([0-9]+) 行,已截断到前 ([0-9]+) 行;建议加 LIMIT 或缩小范围\)", tail)
        if tail and not cap:
            continue
        if not isinstance(rows, list) or not all(isinstance(x, dict) for x in rows):
            continue
        if cap and int(cap.group(2)) != len(rows):
            continue
        parsed.append((rows, bool(cap)))
    if len(parsed) != 1:
        raise IntegrationError("mcp_query_text_invalid")
    return parsed[0]


def _unwrap_mcp_result(value: dict[str, Any]) -> dict[str, Any]:
    status = value.get("status")
    payload = value.get("data")
    if status in {"ok", "success", "not_found", "partial", "unavailable", "error"} and isinstance(payload, dict):
        return {**payload, "status": status, "source_refs": value.get("source_refs", []),
                "code": value.get("code"), "truncated": value.get("truncated", False),
                "as_of": value.get("as_of")}
    return value


__all__ = ["BusinessTools", "IntegrationError"]
