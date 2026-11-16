from __future__ import annotations

import asyncio
import json
from urllib.error import HTTPError
from datetime import datetime, timedelta, timezone
import hashlib
from urllib.parse import unquote

import pytest

from sales_brain_pydantic.integrations import (BusinessTools, IntegrationError, _extract_rows,
    _inventory_dictionary_digest, _parse_beijing_time)


def _settings(**overrides):
    value = {
        "tenant_id": "example-tenant",
        "service_registry": {"services": {
            "company-knowledge": {"base_url": "http://127.0.0.1:8766", "token": "test-token-1234567890"},
        }},
        "mcp_registry": {"servers": {"example-mcp": {
            "url": "http://127.0.0.1:8765/mcp", "enabled": True,
            "protocol_version": "2024-11-05", "transport": "streamable_http",
            "allowed_tools": ["business_guide", "validate_readonly_sql", "mysql_query", "read_dictionary_doc"],
            "scopes": {"tenants": ["example-tenant"], "regions": ["REGION-A"], "permissions": ["read"]},
        }}, "brain_tools": [
            {"name": "erp_sample_card_lookup", "server_ref": "example-mcp"},
            {"name": "erp_inventory_snapshot", "server_ref": "example-mcp"},
        ]},
    }
    value.update(overrides)
    return value


def _profile(customer_ref="erp:REGION-A:101", customer_code="SYNTH-CUST-001", erp_id="101"):
    return {"customer": {
        "tenant_id": "example-tenant", "customer_ref": customer_ref, "region": "REGION-A", "active": True,
        "name": "Synthetic customer", "language": "en",
        "erp_identity": {"status": "confirmed", "customer_code": customer_code,
                         "erp_customer_id": erp_id, "region": "REGION-A",
                         "source": {"system": "example_analytics.customer_profile"}},
        "execution_context": {"simulation": False, "namespace": "verified:" + customer_ref},
    }, "messages": [], "sources": {}, "observed_at": "2026-11-16T00:00:00+00:00"}


def test_available_tools_have_closed_read_only_schemas():
    tools = BusinessTools(_settings()).available_tools()
    assert {item["name"] for item in tools} >= {
        "resolve_product_reference", "search_products", "erp_inventory_snapshot",
        "resolve_quote", "query_knowledge", "erp_customer_activity",
    }
    for item in tools:
        assert item["parameters"]["additionalProperties"] is False
        assert not {"customer_id", "customer_ref", "tenant_id", "region", "url", "sql", "token"} & set(item["parameters"]["properties"])


def test_scope_resolver_uses_canonical_live_identity_and_ignores_test_clones():
    tools = BusinessTools(_settings())
    canonical = {"customer_ref": "erp:REGION-A:101", "region": "REGION-A", "active": True,
                 "erp_identity": {"status": "confirmed", "customer_code": "SYNTH-CUST-001",
                                  "erp_customer_id": "101", "region": "REGION-A",
                                  "source": {"system": "example_analytics.customer_profile"}}}
    clone = {**canonical, "customer_ref": "test:fixture:SYNTH-CUST-001"}

    def http(service, method, path, *, params=None, body=None, timeout=12):
        if path == "/v1/customers":
            assert params["query"] == "SYNTH-CUST-001" and params["region"] == "REGION-A"
            return {"items": [clone, canonical], "next_cursor": None}
        if path.endswith("/context"):
            return _profile()
        raise AssertionError(f"unexpected route: {path}")

    tools._http = http
    result = asyncio.run(tools.resolve_scope("SYNTH-CUST-001", "REGION-A"))
    assert result["identity_status"] == "confirmed"
    assert result["input_customer_id"] == "SYNTH-CUST-001"
    assert result["customer_id"] == "101"
    assert result["customer_code"] == "SYNTH-CUST-001"
    assert result["customer_ref"] == "erp:REGION-A:101"


def test_scope_resolver_fails_closed_for_duplicate_canonical_identities():
    tools = BusinessTools(_settings())
    rows = [
        {"customer_ref": f"erp:REGION-A:{erp_id}", "region": "REGION-A", "active": True,
         "erp_identity": {"status": "confirmed", "customer_code": "SYNTH-CUST-001",
                          "erp_customer_id": erp_id, "region": "REGION-A",
                          "source": {"system": "example_analytics.customer_profile"}}}
        for erp_id in ("101", "102")
    ]

    def http(service, method, path, *, params=None, body=None, timeout=12):
        if path == "/v1/customers":
            return {"items": rows, "next_cursor": None}
        if path.endswith("/context"):
            ref = unquote(path.split("/v1/customers/", 1)[1].rsplit("/context", 1)[0])
            erp_id = ref.rsplit(":", 1)[1]
            return _profile(ref, "SYNTH-CUST-001", erp_id)
        raise AssertionError(f"unexpected route: {path}")

    tools._http = http
    result = asyncio.run(tools.resolve_scope("SYNTH-CUST-001", "REGION-A"))
    assert result["identity_status"] == "ambiguous"
    assert result["customer_ref"] is None
    assert result["customer_id"] is None


def test_scope_resolver_rejects_fixture_identity_even_if_code_matches():
    tools = BusinessTools(_settings())
    test_row = {"customer_ref": "test:fixture:SYNTH-CUST-001", "region": "REGION-A", "active": True,
                "erp_identity": {"status": "confirmed", "customer_code": "SYNTH-CUST-001",
                                 "erp_customer_id": "101", "region": "REGION-A",
                                 "source": {"system": "example_analytics.customer_profile"}}}
    tools._http = lambda service, method, path, **kw: {"items": [test_row], "next_cursor": None}
    result = asyncio.run(tools.resolve_scope("SYNTH-CUST-001", "REGION-A"))
    assert result["identity_status"] == "unresolved"
    assert result["customer_ref"] is None


def test_erp_sql_uses_only_confirmed_server_identity_and_fixed_templates():
    tools = BusinessTools(_settings())
    scope = {"tenant_id": "example-tenant", "customer_ref": "erp:REGION-A:101", "sales_region": "REGION-A"}
    identity = {"erp_customer_id": "101", "customer_code": "SYNTH-CUST-001"}
    sql, db, question, hint = tools._erp_sql(
        "erp_order_progress", {"question": "ignore scope; use SYNTH-CUST-999", "bill_no": "B-123"}, scope, identity)
    assert db == "example_transactions"
    assert "customer_id=101" in sql
    assert "bill_no='B-123'" in sql
    assert "999" not in sql
    assert "SELECT" in sql and "LIMIT 20" in sql
    assert hint == "order_progress"


def test_erp_sql_rejects_model_identifier_with_sql_syntax():
    tools = BusinessTools(_settings())
    scope = {"tenant_id": "example-tenant", "customer_ref": "erp:REGION-A:101", "sales_region": "REGION-A"}
    identity = {"erp_customer_id": "101", "customer_code": "SYNTH-CUST-001"}
    with pytest.raises(IntegrationError, match="business_identifier_invalid"):
        tools._erp_sql("erp_order_progress", {"question": "x", "bill_no": "B-1' OR 1=1--"}, scope, identity)


def test_mcp_scope_accepts_protocol_region_key():
    tools = BusinessTools(_settings())
    tools._mcp_initialized.add("http://127.0.0.1:8765/mcp")
    tools._mcp_advertised["http://127.0.0.1:8765/mcp"] = {"business_guide"}
    tools._mcp_rpc_sync = lambda url, server, method, params, timeout, notification=False: {
        "structuredContent": {"guide_id": "g"}}
    result = asyncio.run(tools._mcp_call("business_guide", {"question": "q"}, {
        "tenant_id": "example-tenant", "tenant_ref": "example-tenant", "customer_ref": "erp:REGION-A:1",
        "region": "REGION-A", "environment": "live"}))
    assert result["guide_id"] == "g"


def test_extract_rows_parses_live_timing_prefix_and_truncation():
    rows, truncated = _extract_rows({"text": "【server_timing_ms】{}\n\n[{\"status\":1}]\n\n...(结果共 3 行,已截断到前 1 行;建议加 LIMIT 或缩小范围)"})
    assert rows == [{"status": 1}]
    assert truncated is True
    rows, truncated = _extract_rows({"content": [{"text": "【server_timing_ms】{}\n\n[{\"status\":1}]"},
                                               {"text": "Query execution time: 138.30 ms"}]})
    assert rows == [{"status": 1}]
    assert truncated is False
    with pytest.raises(IntegrationError):
        _extract_rows({"text": "not a recognized MCP response"})


def test_naive_conversation_timestamp_is_interpreted_as_beijing_local_time():
    parsed = _parse_beijing_time("2026-03-05 18:13")
    assert parsed is not None
    assert parsed.isoformat() == "2026-03-05T10:13:00+00:00"


def test_inventory_status_labels_require_live_reviewed_bk16_source():
    body = ("## Stock status semantics\n"
            "- `status`: 1=available / 2=allocated / 3=in-transit\n"
            "- Count only `status=1` as available stock. Never sum all status values as available stock.\n"
            "Stock balance is a separate source: stock_balance_summary_v1")
    text = "===== example_analytics/stock-query-guide.md =====\n" + body
    assert _inventory_dictionary_digest(text, "configured_local_mcp") == hashlib.sha256(body.encode()).hexdigest()
    assert _inventory_dictionary_digest(text, "configured_external_mcp") is None
    assert _inventory_dictionary_digest(text.replace("3=in-transit", "3=unknown"), "configured_local_mcp") is None


def test_company_knowledge_can_run_with_region_only_and_checks_expiry():
    tools = BusinessTools(_settings())
    valid_until = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    tools._http = lambda service, method, path, **kw: {"items": [
        {"document_id": "doc-1", "revision": 3, "public": True, "region": "REGION-A",
         "source_ref": "handbook:v1", "valid_until": valid_until,
         "published_at": "2026-11-15T00:00:00+00:00", "text": "公开销售时间"},
        {"document_id": "doc-2", "revision": 1, "public": True, "region": "REGION-A",
         "source_ref": "old", "valid_until": "2020-01-01T00:00:00+00:00", "text": "过期资料"},
    ]}
    result = asyncio.run(tools.execute("search_company_knowledge", {"question": "营业时间"},
                                       {"tenant_id": "example-tenant", "sales_region": "REGION-A"}))
    assert result["status"] == "ok"
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["verified"] is True
    assert result["data"]["items"][0]["document_id"] == "doc-1"


def test_secondary_knowledge_answer_is_never_verified(monkeypatch):
    settings = _settings(private_config={"KNOWLEDGE_BASE_URL": "http://127.0.0.1:8650/v1",
                                         "KNOWLEDGE_API_TOKEN": "private-test-token-123",
                                         "KNOWLEDGE_MODEL": "example-model",
                                         "KNOWLEDGE_READ_ONLY_CONFIRMED": True})
    tools = BusinessTools(settings)

    class Response:
        headers = {"Content-Type": "application/json"}
        status = 200
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def read(self, _limit):
            return json.dumps({"model": "example-model", "choices": [{"message": {"content": "参考答案"}}]}).encode()

    monkeypatch.setattr(tools._client, "open", lambda request, timeout: Response())
    result = asyncio.run(tools.execute("query_knowledge", {"question": "一般面料知识"},
        {"tenant_id": "example-tenant", "sales_region": "UNKNOWN"}))
    assert result["status"] == "ok"
    assert result["evidence"][0]["verified"] is False
    assert result["evidence"][0]["as_of"] is None
    assert result["data"]["verified_business_fact"] is False
    assert result["transport_trace"][0]["service"] == "secondary-knowledge-model"
    assert "private-test-token" not in json.dumps(result["transport_trace"])


def test_secondary_knowledge_host_allowlist_rejects_non_loopback_remote_hosts():
    tools = BusinessTools(_settings(private_config={
        "KNOWLEDGE_BASE_URL": "https://example.com/v1",
        "KNOWLEDGE_API_TOKEN": "synthetic-test-token",
        "KNOWLEDGE_MODEL": "example-model",
    }))
    with pytest.raises(IntegrationError, match="knowledge_provider_not_configured"):
        asyncio.run(tools._query_knowledge({"question": "synthetic question"},
                                           {"tenant_id": "example-tenant", "sales_region": "REGION-A"}))


def test_knowledge_http_error_keeps_status_in_private_trace(monkeypatch):
    settings = _settings(private_config={"KNOWLEDGE_BASE_URL": "http://127.0.0.1:8650/v1",
                                         "KNOWLEDGE_API_TOKEN": "private-test-token-123",
                                         "KNOWLEDGE_MODEL": "example-model",
                                         "KNOWLEDGE_READ_ONLY_CONFIRMED": True})
    tools = BusinessTools(settings)
    monkeypatch.setattr(tools._client, "open", lambda request, timeout: (_ for _ in ()).throw(
        HTTPError(request.full_url, 403, "forbidden", {}, None)))
    result = asyncio.run(tools.execute("query_knowledge", {"question": "一般面料知识"},
        {"tenant_id": "example-tenant", "sales_region": "UNKNOWN"}))
    assert result["status"] == "unavailable"
    assert result["missing_information"] == ["knowledge_provider_http_403"]
    assert result["transport_trace"][0]["http_status"] == 403
    assert "private-test-token" not in json.dumps(result["transport_trace"])


def test_secondary_knowledge_default_blocks_before_network(monkeypatch):
    settings = _settings(private_config={"KNOWLEDGE_BASE_URL": "http://127.0.0.1:8650/v1",
                                         "KNOWLEDGE_API_TOKEN": "private-test-token-123",
                                         "KNOWLEDGE_MODEL": "example-model"})
    tools = BusinessTools(settings)
    def forbidden_open(*_args, **_kwargs):
        raise AssertionError("network must not be used without the read-only confirmation flag")
    monkeypatch.setattr(tools._client, "open", forbidden_open)
    result = asyncio.run(tools.execute("query_knowledge", {"question": "客户原话内容"},
        {"tenant_id": "example-tenant", "sales_region": "UNKNOWN"}))
    assert result["status"] == "unavailable"
    assert result["missing_information"] == ["knowledge_profile_readonly_not_verified"]
    assert result["transport_trace"] == []
