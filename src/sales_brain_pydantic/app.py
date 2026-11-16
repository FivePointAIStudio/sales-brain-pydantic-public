from __future__ import annotations

import asyncio
import hashlib
import hmac
import inspect
import json
import shutil
import time
from contextlib import asynccontextmanager
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .agent import AgentRunError, run_sales_agent
from .config import Settings
from .models import ChatRequest
from .store import Store, StoreError, digest


SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_ROOT.parent.parent


async def _wait_for_scope(awaitable, timeout: float):
    return await asyncio.wait_for(awaitable, timeout=timeout)


def _trusted_source_messages(request: ChatRequest, customer_ref: str | None) -> list[dict[str, Any]]:
    """Bind input/history text to server-created refs without treating it as ERP data."""
    if not customer_ref:
        return []
    fence = request.conversation_time_beijing
    rows: list[dict[str, Any]] = []
    if fence is not None:
        for index, item in enumerate(request.history):
            role = {"user": "customer", "assistant": "sales"}.get(item.role)
            if role is None or not item.time_beijing:
                continue
            try:
                created = datetime.fromisoformat(item.time_beijing.replace("Z", "+00:00"))
            except ValueError:
                continue
            if created.tzinfo is None:
                created = created.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
            created = created.astimezone(ZoneInfo("Asia/Shanghai"))
            if created > fence:
                continue
            rows.append({"event_ref": f"input:{request.request_id}:history:{index}",
                         "customer_ref": customer_ref, "role": role, "text": item.text,
                         "created_at": created.isoformat(), "removed": False,
                         "body_truncated": False, "source_revision": 1})
    rows.append({"event_ref": f"input:{request.request_id}:current",
                 "customer_ref": customer_ref, "role": "customer", "text": request.message,
                 "created_at": fence.isoformat() if fence else None,
                 "removed": False, "body_truncated": False, "source_revision": 1})
    return rows


class TakeoverRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    active: bool
    sales_region: str | None = Field(default=None, min_length=2, max_length=32)


def runtime_manifest(settings: Settings) -> dict[str, str]:
    package_root = SOURCE_ROOT
    content = bytearray()
    for path in sorted(package_root.rglob("*.py")):
        if path.is_file():
            content.extend(path.relative_to(package_root).as_posix().encode())
            content.extend(path.read_bytes())
    pyproject = PROJECT_ROOT / "pyproject.toml"
    if pyproject.is_file():
        content.extend(pyproject.read_bytes())
    source_sha = hashlib.sha256(content).hexdigest()
    safe_keys = (
        "OPENAI_MODEL", "OPENAI_BASE_URL", "OPENAI_API_MODE", "OPENAI_REASONING_EFFORT",
        "OPENAI_MAX_OUTPUT_TOKENS", "BRAIN_TOTAL_TIMEOUT_SECONDS", "BRAIN_MODEL_REQUEST_LIMIT",
        "BRAIN_TOOL_CALL_LIMIT", "KNOWLEDGE_BASE_URL", "KNOWLEDGE_MODEL",
        "KNOWLEDGE_READ_ONLY_CONFIRMED", "SA_TENANT_ID",
    )
    effective = {key: settings.get(key) for key in safe_keys if settings.get(key) is not None}
    config_sha = digest(effective)
    try:
        dep_version = package_version("pydantic-ai-slim")
    except PackageNotFoundError:
        dep_version = "unknown"
    runtime_fingerprint = hashlib.sha256(f"{source_sha}:{config_sha}:{dep_version}".encode()).hexdigest()
    environment_identity = {"source_sha": source_sha, "config_sha": config_sha,
                            "runtime_fingerprint": runtime_fingerprint,
                            "dependency_version": dep_version,
                            "knowledge_version": str(settings.get("KNOWLEDGE_MODEL", "unconfigured"))}
    return {**environment_identity, "environment_identity": environment_identity,
            "model": str(settings.get("OPENAI_MODEL"))}


def _customer_key(scope: dict[str, Any], request_id: str) -> str:
    customer_ref = scope.get("customer_ref")
    if isinstance(customer_ref, str) and customer_ref:
        return ":".join(str(scope.get(key) or "") for key in ("tenant_id", "sales_region")) + ":" + customer_ref
    # Anonymous/unresolved requests do not share a synthetic customer identity.
    return "anonymous:" + request_id


def _status_payload(store: Store, run_id: str, current_manifest: dict[str, Any]) -> dict[str, Any]:
    result = store.get_run(run_id)
    if result is None:
        raise HTTPException(404, "run_not_found")
    saved_manifest = result.pop("runtime_manifest", None)
    if isinstance(saved_manifest, dict):
        result.update(saved_manifest)
    else:
        unknown = {"source_sha": "unknown", "config_sha": "unknown",
                   "runtime_fingerprint": "unknown", "dependency_version": "unknown",
                   "knowledge_version": "unknown", "model": "unknown",
                   "environment_identity": {"source_sha": "unknown", "config_sha": "unknown",
                                            "runtime_fingerprint": "unknown", "dependency_version": "unknown",
                                            "knowledge_version": "unknown", "version_status": "legacy_unknown"}}
        result.update(unknown)
    result["current_runtime_fingerprint"] = current_manifest["runtime_fingerprint"]
    return result


def create_app(*, settings: Settings | None = None,
               store: Store | None = None,
               business_tools_factory=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            yield
        finally:
            application.state.store.close()

    app = FastAPI(title="Sales Brain PydanticAI", version=__version__, lifespan=lifespan)
    app.state.settings = settings or Settings.load()
    app.state.store = store or Store(app.state.settings.database_path)
    app.state.business_tools_factory = business_tools_factory
    # Snapshot the resolved source tree and non-secret config once for this process.
    app.state.runtime_manifest = runtime_manifest(app.state.settings)

    async def authenticated(authorization: str | None = Header(default=None)) -> None:
        expected = app.state.settings.get("BRAIN_API_TOKEN")
        if not isinstance(expected, str) or not expected:
            raise HTTPException(503, "service_auth_not_configured")
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "bearer_token_required")
        provided = authorization[7:].strip()
        if not hmac.compare_digest(provided, expected):
            raise HTTPException(401, "invalid_bearer_token")

    async def make_business_tools():
        factory = app.state.business_tools_factory
        if factory is None:
            from .integrations import BusinessTools
            factory = BusinessTools
        value = factory(app.state.settings)
        if inspect.isawaitable(value):
            value = await value
        return value

    async def close_business_tools(value: Any) -> None:
        close = getattr(value, "aclose", None)
        if close:
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass

    async def resolve_scope(business_tools: Any, request: ChatRequest) -> tuple[dict[str, Any], str | None]:
        resolver = getattr(business_tools, "resolve_scope", None)
        if resolver is None:
            raise HTTPException(503, "business_scope_resolver_unavailable")
        try:
            scope = resolver(request.customer_id, request.sales_region)
            if inspect.isawaitable(scope):
                scope = await scope
        except Exception:
            # Fail closed for private customer facts; region-only public lookup can continue.
            return {"tenant_id": str(app.state.settings.get("SA_TENANT_ID", "example-tenant")),
                    "customer_id": None, "customer_ref": None,
                    "sales_region": request.sales_region, "resolution_status": "unavailable"}, None
        if not isinstance(scope, dict):
            raise HTTPException(503, "business_scope_contract_invalid")
        scope = dict(scope)
        scope.setdefault("input_customer_id", request.customer_id)
        if request.sales_region and scope.get("sales_region") and request.sales_region != scope["sales_region"]:
            return {"tenant_id": scope.get("tenant_id"), "customer_id": None, "customer_ref": None,
                    "sales_region": request.sales_region, "resolution_status": "region_conflict"}, None
        if request.customer_id and scope.get("customer_ref"):
            matched_input = scope.get("input_customer_id")
            if matched_input is None and scope.get("customer_code") == request.customer_id:
                matched_input = scope.get("customer_code")
            if str(matched_input) != str(request.customer_id):
                return {"tenant_id": scope.get("tenant_id"), "customer_id": None, "customer_ref": None,
                        "sales_region": request.sales_region, "resolution_status": "identity_mismatch"}, None
        if not scope.get("customer_ref"):
            # Unresolved customer identifiers must not cross the integration boundary as authority.
            scope["customer_id"] = None
        scope["source_messages"] = _trusted_source_messages(request, scope.get("customer_ref"))
        scope["reply_language"] = request.reply_language
        scope["conversation_time_beijing"] = (
            request.conversation_time_beijing.isoformat() if request.conversation_time_beijing else None
        )
        return scope, scope.get("customer_ref") if isinstance(scope.get("customer_ref"), str) else None

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"service": "sales-brain-pydantic-public", "status": "ready", **app.state.runtime_manifest}

    @app.get("/v1/status", dependencies=[Depends(authenticated)])
    def status() -> dict[str, Any]:
        return {"service": "sales-brain-pydantic-public", "status": "ready",
                "execution_mode": "proposal_only", "external_business_writes": False,
                **app.state.runtime_manifest}

    @app.post("/v1/chat", dependencies=[Depends(authenticated)])
    async def chat(request: ChatRequest) -> dict[str, Any]:
        settings = app.state.settings
        store = app.state.store
        body = request.model_dump(mode="json")
        try:
            existing = store.lookup_request(request.request_id, body)
        except StoreError as exc:
            raise HTTPException(409, exc.code) from None
        if existing:
            return _status_payload(store, existing["run_id"], app.state.runtime_manifest)

        # Keep room for the service and future audit writes. Existing requests remain readable above.
        free_bytes = shutil.disk_usage(settings.database_path.parent).free
        if free_bytes < 3 * 1024 ** 3:
            raise HTTPException(503, "disk_space_below_3gib_model_gate")

        started = time.monotonic()
        deadline = started + int(settings.get("BRAIN_TOTAL_TIMEOUT_SECONDS", 240))
        try:
            arrival_seq = store.reserve_ingress(request.request_id, body)
        except StoreError as exc:
            raise HTTPException(409, exc.code) from None
        business_tools = await make_business_tools()
        try:
            return await process_chat(request, body, business_tools, arrival_seq, deadline)
        finally:
            await close_business_tools(business_tools)

    async def process_chat(request: ChatRequest, body: dict[str, Any], business_tools: Any,
                           arrival_seq: int, deadline: float) -> dict[str, Any]:
        settings = app.state.settings
        store = app.state.store
        try:
            scope, customer_ref = await _wait_for_scope(
                resolve_scope(business_tools, request), timeout=max(1, deadline - time.monotonic())
            )
        except asyncio.TimeoutError:
            raise HTTPException(504, "customer_scope_timeout") from None
        run_id = "run_" + uuid4().hex
        customer_key = _customer_key(scope, request.request_id)
        try:
            run, created = store.create_run(run_id=run_id, request_id=request.request_id,
                                            body=body, customer_key=customer_key,
                                            customer_ref=customer_ref, scope=scope,
                                            manifest=app.state.runtime_manifest,
                                            arrival_seq=arrival_seq)
        except StoreError as exc:
            raise HTTPException(409, exc.code) from None
        if not created:
            return _status_payload(store, run["run_id"], app.state.runtime_manifest)
        if run["status"] == "stale":
            return _status_payload(store, run_id, app.state.runtime_manifest)
        scope["request_id"] = request.request_id
        scope["run_id"] = run_id

        if store.manual_takeover(customer_key):
            blocked = {"status": "blocked", "reply": "", "actions": [], "evidence_refs": [],
                       "missing_information": ["manual_takeover"], "usage": {},
                       "tool_trace": [], "model_messages": [], "evidence": []}
            store.save_result(run_id, customer_key, blocked)
            return _status_payload(store, run_id, app.state.runtime_manifest)

        got_lock = await asyncio.to_thread(
            store.acquire_customer_lock, run_id=run_id, customer_key=customer_key,
            wait_seconds=max(0, deadline - time.monotonic()),
            lease_seconds=max(30, int(settings.get("BRAIN_TOTAL_TIMEOUT_SECONDS", 240)) + 60),
        )
        if not got_lock:
            store.finish_error(run_id, {"code": "customer_run_wait_timeout"}, status="cancelled")
            return _status_payload(store, run_id, app.state.runtime_manifest)

        try:
            if store.manual_takeover(customer_key):
                blocked = {"status": "blocked", "reply": "", "actions": [], "evidence_refs": [],
                           "missing_information": ["manual_takeover"], "usage": {},
                           "tool_trace": [], "model_messages": [], "evidence": []}
                store.save_result(run_id, customer_key, blocked)
            elif time.monotonic() >= deadline:
                store.finish_error(run_id, {"code": "run_deadline"}, status="failed")
            else:
                try:
                    result = await asyncio.wait_for(
                        run_sales_agent(request=request, settings=settings, store=store,
                                        business_tools=business_tools, scope=scope,
                                        run_id=run_id, customer_key=customer_key,
                                        deadline=deadline),
                        timeout=max(0.1, deadline - time.monotonic()),
                    )
                    store.save_result(run_id, customer_key, result)
                except AgentRunError as exc:
                    store.record_evidence(run_id, exc.audit.get("evidence", []))
                    store.finish_error(run_id, {"code": exc.code}, audit=exc.audit)
                except Exception as exc:
                    code = getattr(exc, "code", None) or ("run_timeout" if isinstance(exc, asyncio.TimeoutError) else "planning_failed")
                    store.finish_error(run_id, {"code": str(code)}, audit={"tool_trace": [], "model_messages": [], "evidence": []})
        finally:
            store.release_customer_lock(run_id=run_id, customer_key=customer_key)
        return _status_payload(store, run_id, app.state.runtime_manifest)

    @app.get("/v1/runs/{run_id}", dependencies=[Depends(authenticated)])
    def read_run(run_id: str) -> dict[str, Any]:
        return _status_payload(app.state.store, run_id, app.state.runtime_manifest)

    @app.post("/v1/runs/{run_id}/cancel", dependencies=[Depends(authenticated)])
    def cancel_run(run_id: str) -> dict[str, Any]:
        if app.state.store.cancel_run(run_id) is None:
            raise HTTPException(404, "run_not_found")
        return _status_payload(app.state.store, run_id, app.state.runtime_manifest)

    @app.put("/v1/customers/{customer_id}/takeover", dependencies=[Depends(authenticated)])
    async def set_takeover(customer_id: str, body: TakeoverRequest) -> dict[str, Any]:
        business_tools = await make_business_tools()
        try:
            request = ChatRequest(request_id="takeover_" + uuid4().hex,
                                  customer_id=customer_id, sales_region=body.sales_region,
                                  message="takeover control", conversation_time_beijing=None)
            scope, customer_ref = await resolve_scope(business_tools, request)
            if not customer_ref:
                raise HTTPException(422, "customer_identity_not_resolved")
            customer_key = _customer_key(scope, request.request_id)
            app.state.store.set_manual_takeover(customer_key, body.active)
            return {"customer_ref": customer_ref, "manual_takeover": body.active,
                    "guard": "new_runs_blocked" if body.active else "new_runs_allowed"}
        finally:
            await close_business_tools(business_tools)

    return app


app = create_app()

