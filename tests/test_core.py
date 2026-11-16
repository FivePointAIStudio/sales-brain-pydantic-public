from __future__ import annotations

import json
import os
import asyncio
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest
import httpx
from fastapi.testclient import TestClient

from sales_brain_pydantic.agent import SYSTEM_INSTRUCTIONS, _prompt, build_model
from sales_brain_pydantic import agent as agent_module
from sales_brain_pydantic.config import Settings
from sales_brain_pydantic.models import ChatRequest
from sales_brain_pydantic.store import Store, StoreError


def test_synthetic_input_contract_validates_without_evaluation_labels():
    row = {"input": {"request_id": "synthetic_request_01", "customer_id": None,
                     "sales_region": "REGION-A", "reply_language": "English",
                     "conversation_time_beijing": "2026-10-08 09:54",
                     "history": [], "message": "Please check this synthetic item."},
           "expected": {"private_label": "not-for-model-input"}, "case_id": "SYNTH-001"}
    request = ChatRequest.model_validate(row["input"])
    assert "case_id" not in request.model_dump()
    assert "expected" not in request.model_dump()



def test_synthetic_time_region_and_participant_are_normalized():
    request = ChatRequest.model_validate({
        "request_id": "request_12345678",
        "customer_id": None,
        "sales_region": "Region A",
        "reply_language": "English",
        "conversation_time_beijing": "2026-10-08 09:54",
        "history": [{"role": "participant", "original_speaker_type": "会话成员",
                     "text": "当前信息已核实", "time_beijing": "2026-10-08 09:40"}],
        "message": "Please check this synthetic item.",
    })
    assert request.sales_region == "REGION-A"
    assert request.conversation_time_beijing.isoformat() == "2026-10-08T09:54:00+08:00"
    assert request.history[0].role == "participant"
    assert request.history[0].text == "当前信息已核实"


def test_null_conversation_time_is_not_replaced_with_now():
    request = ChatRequest.model_validate({
        "request_id": "request_12345679", "customer_id": None, "sales_region": None,
        "reply_language": "zh-CN", "conversation_time_beijing": None,
        "history": [], "message": "你好",
    })
    assert request.conversation_time_beijing is None


def test_scope_source_messages_are_server_bound_and_time_fenced():
    from sales_brain_pydantic.app import _trusted_source_messages

    request = ChatRequest.model_validate({
        "request_id": "source_fence_123456", "customer_id": "SYNTH-CUST-001", "sales_region": "REGION-A",
        "reply_language": "English", "conversation_time_beijing": "2026-10-08 10:00",
        "history": [
            {"role": "user", "original_speaker_type": "客户", "text": "red fabric", "time_beijing": "2026-10-08 09:00"},
            {"role": "assistant", "original_speaker_type": "销售", "text": "I will check", "time_beijing": "2026-10-08 09:10"},
            {"role": "participant", "original_speaker_type": "会话成员", "text": "biru", "time_beijing": "2026-10-08 09:20"},
            {"role": "user", "original_speaker_type": "客户", "text": "green fabric", "time_beijing": "2026-10-08 11:00"},
        ],
        "message": "Please check red fabric",
    })
    messages = _trusted_source_messages(request, "erp:REGION-A:123")
    assert [item["role"] for item in messages] == ["customer", "sales", "customer"]
    assert [item["text"] for item in messages] == ["red fabric", "I will check", "Please check red fabric"]
    assert all(item["customer_ref"] == "erp:REGION-A:123" for item in messages)
    assert messages[-1]["created_at"] == "2026-10-08T10:00:00+08:00"


def test_system_guidance_separates_current_fact_and_hypothetical_policy_questions():
    examples = [
        "What is the current price of SYNTH-PROD-001?",
        "Can the color change after an order is placed?",
        "Please change the color on my existing order.",
    ]
    for index, message in enumerate(examples):
        request = ChatRequest.model_validate({
            "request_id": f"guidance_{index}_123456", "customer_id": None, "sales_region": "Region A",
            "reply_language": "English", "conversation_time_beijing": "2026-10-08 09:54",
            "history": [], "message": message,
        })
        prompt = json.loads(_prompt(request, None, "message-evidence"))
        assert prompt["current_customer_message"] == message
        assert "conversation_time_beijing" in prompt
    assert "does not by itself turn an ordinary stock or price question into a request for historical auditing" in SYSTEM_INSTRUCTIONS
    assert "hypothetical" in SYSTEM_INSTRUCTIONS and "actual order" in SYSTEM_INSTRUCTIONS

    referential = ChatRequest.model_validate({
        "request_id": "guidance_ref_123456", "customer_id": None, "sales_region": "REGION-A",
        "reply_language": "English", "conversation_time_beijing": "2026-10-08 09:54",
        "history": [{"role": "assistant", "original_speaker_type": "销售侧",
                     "text": "The sales team received SYNTH-ORDER-001.", "time_beijing": "2026-10-08 09:40"}],
        "message": "Can the color on SYNTH-ORDER-001 be changed?",
    })
    referential_prompt = json.loads(_prompt(referential, None, "message-evidence"))
    assert "The sales team received SYNTH-ORDER-001." in referential_prompt["history"][0]["text"]
    assert "reuse its known identifier" in SYSTEM_INSTRUCTIONS
    assert "do not by themselves make a question hypothetical" in SYSTEM_INSTRUCTIONS


def test_settings_use_responses_for_reasoning_model():
    settings = Settings({"OPENAI_API_KEY": "test-secret", "BRAIN_API_TOKEN": "test-token"})
    model = build_model(settings)
    from pydantic_ai.models.openai import OpenAIResponsesModel
    assert isinstance(model, OpenAIResponsesModel)
    assert model.model_name == "example-model"


def test_chat_mode_does_not_silently_drop_reasoning_effort():
    with pytest.raises(ValueError, match="chat_api_requires_reasoning_effort_none"):
        Settings({"OPENAI_API_MODE": "chat", "OPENAI_REASONING_EFFORT": "medium"})


def test_pydanticai_dynamic_tool_loop_records_actual_messages_and_results(tmp_path, monkeypatch):
    from pydantic_ai.models.test import TestModel

    class OneTool:
        calls = 0

        def available_tools(self):
            return [{"name": "read_customer_context", "description": "Read verified customer context.",
                     "parameters": {"type": "object", "properties": {}, "required": [],
                                    "additionalProperties": False}}]

        async def execute(self, name, args, scope):
            assert name == "read_customer_context"
            assert args == {}
            self.calls += 1
            if self.calls == 2:
                class TransportFailure(RuntimeError):
                    code = "fixture_remote_error"
                    transport_trace = [{"transport": "mcp", "tool": "mysql_query",
                                        "request": {"sql": "SELECT 1"},
                                        "result": {"status": "remote_error", "marker": "transport_marker_error"}}]
                raise TransportFailure("remote failure")
            return {"status": "ok", "data": {"verified": True},
                    "evidence": [{"evidence_ref": "ev-context", "source": "customer-context",
                                  "source_ref": "customer:rev-2", "as_of": "2026-11-16T00:00:00+08:00",
                                  "customer_ref": "erp:REGION-A:123", "region": "REGION-A", "verified": True}],
                    "missing_information": [],
                    "transport_trace": [{"transport": "http", "service": "customer-context",
                                         "method": "GET", "request": {"path": "/v1/customers/x/context"},
                                         "headers": {"Authorization": "Bearer hidden"},
                                         "result": {"status_code": 200, "marker": "transport_marker_ok"}}]}

    settings = Settings({"BRAIN_STATE_DIR": str(tmp_path), "OPENAI_API_KEY": "unused"})
    store = Store(tmp_path / "brain.sqlite3")
    request_id = "request_tool_loop_123"
    request = ChatRequest.model_validate({
        "request_id": request_id, "customer_id": "SYNTH-CUST-001", "sales_region": "REGION-A",
        "reply_language": "English", "conversation_time_beijing": datetime.now().astimezone().isoformat(),
        "history": [], "message": "Tolong cek ya.",
    })
    run_id = "run_tool_loop"
    customer_key = "tenant-a:REGION-A:erp:REGION-A:123"
    store.create_run(run_id=run_id, request_id=request_id, body=request.model_dump(mode="json"),
                     customer_key=customer_key, customer_ref="erp:REGION-A:123",
                     scope={"tenant_id": "tenant-a", "customer_id": "123", "customer_ref": "erp:REGION-A:123",
                            "customer_code": "SYNTH-CUST-001", "sales_region": "REGION-A"})
    store.acquire_customer_lock(run_id=run_id, customer_key=customer_key, wait_seconds=0)
    fake_model = TestModel(call_tools=["read_customer_context"], custom_output_args={
        "status": "ready", "reply": "Saya bantu cek.", "actions": [],
        "evidence_refs": ["ev-context"], "missing_information": [],
    })
    monkeypatch.setattr(agent_module, "build_model", lambda _: fake_model)
    result = asyncio.run(agent_module.run_sales_agent(
        request=request, settings=settings, store=store, business_tools=OneTool(),
        scope={"tenant_id": "tenant-a", "customer_id": "123", "customer_code": "SYNTH-CUST-001",
               "customer_ref": "erp:REGION-A:123", "sales_region": "REGION-A"},
        run_id=run_id, customer_key=customer_key, deadline=time.monotonic() + 30,
    ))
    assert result["reply"] == "Saya bantu cek."
    assert len(result["tool_trace"]) == 2  # deterministic context read plus model-called read
    assert all(item["run_id"] == run_id for item in result["tool_trace"])
    assert all(item["actual_arguments"] == {} for item in result["tool_trace"])
    assert result["tool_trace"][0]["result"]["evidence"][0]["source_ref"] == "customer:rev-2"
    assert result["tool_trace"][0]["transport_trace"][0]["result"]["marker"] == "transport_marker_ok"
    assert "headers" not in result["tool_trace"][0]["transport_trace"][0]
    assert result["tool_trace"][1]["transport_trace"][0]["result"]["marker"] == "transport_marker_error"
    assert any(message.get("kind") == "response" for message in result["model_messages"])
    assert "transport_marker" not in json.dumps(result["model_messages"], ensure_ascii=False)
    assert result["evidence_refs"] == ["ev-context"]


def test_store_binds_idempotency_and_stales_older_actions(tmp_path):
    store = Store(tmp_path / "brain.sqlite3")
    scope = {"tenant_id": "tenant-a", "customer_ref": "erp:REGION-A:123",
             "sales_region": "REGION-A", "customer_id": "123", "customer_code": "SYNTH-CUST-001"}
    first_body = {"request_id": "request_12345678", "message": "need a quote"}
    first, created = store.create_run(run_id="run-one", request_id=first_body["request_id"],
                                      body=first_body, customer_key="tenant-a:REGION-A:erp:REGION-A:123",
                                      customer_ref=scope["customer_ref"], scope=scope)
    assert created and first["status"] == "queued"
    assert store.acquire_customer_lock(run_id="run-one", customer_key="tenant-a:REGION-A:erp:REGION-A:123", wait_seconds=0)
    store.save_result("run-one", "tenant-a:REGION-A:erp:REGION-A:123", {
        "status": "ready", "reply": "I can help check this.",
        "actions": [{"kind": "quote_request", "summary": "Check the current quote",
                     "evidence_refs": ["ev-1"], "execution_status": "not_executed"}],
        "evidence_refs": ["ev-1"], "missing_information": [],
    })
    store.release_customer_lock(run_id="run-one", customer_key="tenant-a:REGION-A:erp:REGION-A:123")
    assert store.get_run("run-one")["execution_receipts"][0]["status"] == "planned"
    assert store.get_run("run-one")["execution_receipts"][0]["execution_status"] == "not_executed"

    second_body = {"request_id": "request_87654321", "message": "newer customer message"}
    store.create_run(run_id="run-two", request_id=second_body["request_id"], body=second_body,
                     customer_key="tenant-a:REGION-A:erp:REGION-A:123",
                     customer_ref=scope["customer_ref"], scope=scope)
    old = store.get_run("run-one")
    assert old["status"] == "stale"
    assert old["reply"] == ""
    assert old["actions"] == []
    assert old["execution_receipts"][0]["status"] == "stale"

    with pytest.raises(StoreError, match="idempotency_conflict"):
        store.lookup_request(first_body["request_id"], {**first_body, "message": "different content"})


def test_restart_marks_inflight_runs_interrupted_and_clears_old_locks(tmp_path):
    path = tmp_path / "brain.sqlite3"
    store = Store(path)
    scope = {"tenant_id": "tenant-a", "customer_ref": "erp:REGION-A:123", "sales_region": "REGION-A"}
    run, _ = store.create_run(run_id="run-before-restart", request_id="request_before_restart",
                              body={"request_id": "request_before_restart"},
                              customer_key="tenant-a:REGION-A:erp:REGION-A:123",
                              customer_ref=scope["customer_ref"], scope=scope)
    assert store.acquire_customer_lock(run_id=run["run_id"],
                                       customer_key="tenant-a:REGION-A:erp:REGION-A:123", wait_seconds=0)
    with pytest.raises(StoreError, match="state_directory_in_use"):
        Store(path)
    store.close()

    restarted = Store(path)
    previous = restarted.get_run("run-before-restart")
    assert previous["status"] == "interrupted"
    assert previous["error"]["code"] == "process_interrupted"
    new, _ = restarted.create_run(run_id="run-after-restart", request_id="request_after_restart",
                                  body={"request_id": "request_after_restart"},
                                  customer_key="tenant-a:REGION-A:erp:REGION-A:123",
                                  customer_ref=scope["customer_ref"], scope=scope)
    assert restarted.acquire_customer_lock(run_id=new["run_id"],
                                           customer_key="tenant-a:REGION-A:erp:REGION-A:123", wait_seconds=0)
    restarted.release_customer_lock(run_id=new["run_id"], customer_key="tenant-a:REGION-A:erp:REGION-A:123")
    restarted.close()


def test_cancel_terminal_wins_over_late_error_or_success(tmp_path):
    store = Store(tmp_path / "brain.sqlite3")
    scope = {"tenant_id": "tenant-a", "customer_ref": "erp:REGION-A:123", "sales_region": "REGION-A"}
    run, _ = store.create_run(run_id="run-terminal", request_id="request_terminal_123",
                              body={"request_id": "request_terminal_123"},
                              customer_key="tenant-a:REGION-A:erp:REGION-A:123",
                              customer_ref=scope["customer_ref"], scope=scope)
    assert store.cancel_run(run["run_id"])["status"] == "cancelled"
    store.finish_error(run["run_id"], {"code": "late_provider_error"})
    assert not store.save_result(run["run_id"], "tenant-a:REGION-A:erp:REGION-A:123", {
        "status": "ready", "reply": "must never be returned", "actions": [], "evidence_refs": [],
    })
    result = store.get_run(run["run_id"])
    assert result["status"] == "cancelled"
    assert result["reply"] == ""
    assert result["error"] is None or result["error"].get("code") != "late_provider_error"


def test_old_run_keeps_its_saved_manifest(tmp_path):
    store = Store(tmp_path / "brain.sqlite3")
    manifest_a = {"source_sha": "source-a", "config_sha": "config-a",
                  "runtime_fingerprint": "runtime-a", "model": "model-a"}
    run, _ = store.create_run(run_id="run-manifest", request_id="request_manifest_1",
                              body={"request_id": "request_manifest_1"},
                              customer_key="anonymous:request_manifest_1", customer_ref=None,
                              manifest=manifest_a, arrival_seq=1)
    assert store.get_run(run["run_id"])["runtime_manifest"] == manifest_a
    legacy, _ = store.create_run(run_id="run-legacy", request_id="request_manifest_2",
                                 body={"request_id": "request_manifest_2"},
                                 customer_key="anonymous:request_manifest_2", customer_ref=None,
                                 manifest=None, arrival_seq=2)
    from sales_brain_pydantic.app import _status_payload
    payload = _status_payload(store, legacy["run_id"], {"runtime_fingerprint": "current-b"})
    assert payload["source_sha"] == "unknown"
    assert payload["environment_identity"]["version_status"] == "legacy_unknown"
    assert payload["current_runtime_fingerprint"] == "current-b"


def test_cancel_and_handoff_hide_saved_candidate_actions(tmp_path):
    store = Store(tmp_path / "brain.sqlite3")
    scope = {"tenant_id": "tenant-a", "customer_ref": "erp:REGION-A:5", "sales_region": "REGION-A"}
    run, _ = store.create_run(run_id="run-cancel", request_id="request_cancel_1",
                              body={"request_id": "request_cancel_1"},
                              customer_key="tenant-a:REGION-A:erp:REGION-A:5",
                              customer_ref=scope["customer_ref"], scope=scope)
    store.acquire_customer_lock(run_id=run["run_id"], customer_key="tenant-a:REGION-A:erp:REGION-A:5", wait_seconds=0)
    store.save_result(run["run_id"], "tenant-a:REGION-A:erp:REGION-A:5", {
        "status": "ready", "reply": "Candidate reply", "actions": [], "evidence_refs": [],
    })
    store.set_manual_takeover("tenant-a:REGION-A:erp:REGION-A:5", True)
    result = store.get_run(run["run_id"])
    assert result["status"] in {"cancelled", "stale"}
    assert result["reply"] == ""
    assert result["actions"] == []


class FakeBusinessTools:
    async def resolve_scope(self, customer_id, sales_region):
        return {"tenant_id": "tenant-a", "input_customer_id": customer_id,
                "customer_id": "123", "customer_code": customer_id,
                "customer_ref": "erp:REGION-A:123", "sales_region": sales_region or "REGION-A",
                "identity_status": "confirmed"}

    def available_tools(self):
        return []


def test_api_requires_bearer_and_records_actions_as_not_executed(tmp_path, monkeypatch):
    from sales_brain_pydantic import app as app_module

    settings = Settings({"BRAIN_API_TOKEN": "test-token", "BRAIN_STATE_DIR": str(tmp_path),
                         "OPENAI_MODEL": "example-model"})
    app = app_module.create_app(settings=settings, store=Store(tmp_path / "api.sqlite3"),
                                business_tools_factory=lambda _: FakeBusinessTools())

    async def fake_run_sales_agent(**kwargs):
        return {"status": "ready", "reply": "Saya akan bantu mengeceknya.",
                "actions": [{"kind": "quote_request", "summary": "Check a current quote",
                             "parameters": {}, "evidence_refs": ["message-ev"],
                             "preconditions": [], "status": "proposed",
                             "execution_status": "not_executed"}],
                "evidence_refs": ["message-ev"], "missing_information": [],
                "usage": {}, "tool_trace": [], "model_messages": [],
                "evidence": [{"evidence_ref": "message-ev", "source": "current_customer_message"}]}

    monkeypatch.setattr(app_module, "run_sales_agent", fake_run_sales_agent)
    request = {"request_id": "request_api_12345", "customer_id": "SYNTH-CUST-001",
               "sales_region": "Region A", "reply_language": "English",
               "conversation_time_beijing": datetime.now().astimezone().isoformat(),
               "history": [], "message": "Can you check this?"}
    with TestClient(app) as client:
        assert client.post("/v1/chat", json=request).status_code == 401
        response = client.post("/v1/chat", json=request,
                               headers={"Authorization": "Bearer test-token"})
        assert response.status_code == 200
        saved_source_sha = response.json()["source_sha"]
        app.state.runtime_manifest = {**app.state.runtime_manifest,
                                      "source_sha": "new-current-source",
                                      "runtime_fingerprint": "new-current-runtime"}
        old_run = client.get(f"/v1/runs/{response.json()['run_id']}",
                             headers={"Authorization": "Bearer test-token"}).json()
    assert response.status_code == 200
    body = response.json()
    assert body["customer_scope"]["verified"] is True
    assert body["runtime_fingerprint"]
    assert body["environment_identity"]["source_sha"] == body["source_sha"]
    assert body["actions"][0]["execution_status"] == "not_executed"
    assert body["execution_receipts"][0]["status"] == "planned"
    assert body["execution_receipts"][0]["external_effect"] is False
    assert old_run["source_sha"] == saved_source_sha
    assert old_run["current_runtime_fingerprint"] == "new-current-runtime"


def test_ingress_order_stops_slow_older_scope_from_staling_newer_message(tmp_path, monkeypatch):
    from sales_brain_pydantic import app as app_module

    close_counts: dict[str, int] = {}
    a_started = asyncio.Event()
    release_a = asyncio.Event()

    class DelayedTools:
        def __init__(self):
            self.label = "unknown"

        async def resolve_scope(self, customer_id, sales_region):
            self.label = customer_id
            if customer_id == "A":
                a_started.set()
                await release_a.wait()
            return {"tenant_id": "tenant-a", "input_customer_id": customer_id,
                    "customer_id": "123", "customer_code": customer_id,
                    "customer_ref": "erp:REGION-A:123", "sales_region": "REGION-A",
                    "identity_status": "confirmed"}

        def available_tools(self):
            return []

        async def aclose(self):
            close_counts[self.label] = close_counts.get(self.label, 0) + 1

    settings = Settings({"BRAIN_API_TOKEN": "token", "BRAIN_STATE_DIR": str(tmp_path),
                         "OPENAI_MODEL": "example-model"})
    app = app_module.create_app(settings=settings, store=Store(tmp_path / "arrival.sqlite3"),
                                business_tools_factory=lambda _: DelayedTools())

    async def fake_run_sales_agent(**kwargs):
        return {"status": "ready", "reply": "Reply for the latest accepted message.",
                "actions": [], "evidence_refs": [], "missing_information": [], "usage": {},
                "tool_trace": [], "model_messages": [], "evidence": []}

    monkeypatch.setattr(app_module, "run_sales_agent", fake_run_sales_agent)

    async def send(client, label):
        payload = {"request_id": "request_arrival_" + label,
                   "customer_id": label, "sales_region": "REGION-A", "reply_language": "English",
                   "conversation_time_beijing": "2026-10-09 10:00" if label == "A" else "2026-10-08 10:00",
                   "history": [], "message": "message " + label}
        return await client.post("/v1/chat", json=payload,
                                 headers={"Authorization": "Bearer token"})

    async def scenario():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                older = asyncio.create_task(send(client, "A"))
                await a_started.wait()  # A has received the first ingress sequence, then blocks in scope.
                newer = await send(client, "B")
                release_a.set()
                older = await older
                return older.json(), newer.json()

    old_result, new_result = asyncio.run(scenario())
    assert old_result["status"] == "stale"
    assert old_result["reply"] == ""
    assert new_result["status"] == "ready"
    assert new_result["reply"] == "Reply for the latest accepted message."
    assert close_counts == {"A": 1, "B": 1}


def test_business_tools_close_once_after_scope_timeout(tmp_path, monkeypatch):
    from sales_brain_pydantic import app as app_module

    close_count = 0

    class Tools(FakeBusinessTools):
        async def aclose(self):
            nonlocal close_count
            close_count += 1

    settings = Settings({"BRAIN_API_TOKEN": "token", "BRAIN_STATE_DIR": str(tmp_path)})
    app = app_module.create_app(settings=settings, store=Store(tmp_path / "timeout.sqlite3"),
                                business_tools_factory=lambda _: Tools())

    async def force_timeout(awaitable, timeout):
        close = getattr(awaitable, "close", None)
        if close:
            close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(app_module, "_wait_for_scope", force_timeout)
    payload = {"request_id": "request_scope_timeout", "customer_id": "SYNTH-CUST-001",
               "sales_region": "REGION-A", "reply_language": "English",
               "conversation_time_beijing": None, "history": [], "message": "check"}
    with TestClient(app) as client:
        response = client.post("/v1/chat", json=payload,
                               headers={"Authorization": "Bearer token"})
    assert response.status_code == 504
    assert close_count == 1


def test_business_tools_close_once_on_manual_takeover_and_lock_timeout(tmp_path, monkeypatch):
    from sales_brain_pydantic import app as app_module

    settings = Settings({"BRAIN_API_TOKEN": "token", "BRAIN_STATE_DIR": str(tmp_path)})
    store = Store(tmp_path / "guards.sqlite3")
    key = "tenant-a:REGION-A:erp:REGION-A:123"
    store.set_manual_takeover(key, True)
    close_counts = {"takeover": 0, "lock": 0}

    class Tools(FakeBusinessTools):
        def __init__(self, label):
            self.label = label

        async def aclose(self):
            close_counts[self.label] += 1

    current = "takeover"
    app = app_module.create_app(settings=settings, store=store,
                                business_tools_factory=lambda _: Tools(current))
    payload = {"request_id": "request_guard_takeover", "customer_id": "SYNTH-CUST-001",
               "sales_region": "REGION-A", "reply_language": "English",
               "conversation_time_beijing": None, "history": [], "message": "check"}
    with TestClient(app) as client:
        takeover = client.post("/v1/chat", json=payload,
                               headers={"Authorization": "Bearer token"})
        assert takeover.status_code == 200 and takeover.json()["status"] == "blocked"

        store.set_manual_takeover(key, False)
        current = "lock"
        monkeypatch.setattr(store, "acquire_customer_lock", lambda **_: False)
        payload["request_id"] = "request_guard_lock_timeout"
        lock_timeout = client.post("/v1/chat", json=payload,
                                   headers={"Authorization": "Bearer token"})
    assert lock_timeout.status_code == 200
    assert lock_timeout.json()["status"] == "cancelled"
    assert close_counts == {"takeover": 1, "lock": 1}

