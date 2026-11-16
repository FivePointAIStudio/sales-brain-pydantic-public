# Sales Brain PydanticAI

A reference service for evidence-bound sales assistance. The package accepts a conversation request, resolves trusted context through configured read-only adapters, and returns a proposed reply with evidence, missing information, and proposed actions. It records proposals locally; it does not send messages or perform business writes.

This is a software reference implementation. It has not been validated against any organization’s live systems, business data, policies, or production workload. An integrator must provide and review their own service registries, MCP server, identity mapping, read-only data adapters, authentication secrets, and deployment controls before connected use.

## Capabilities and boundaries

- FastAPI endpoints for health, authenticated status and chat, run status/cancellation, and manual takeover controls.
- PydanticAI model integration and bounded model/tool calls.
- Evidence references and trusted-scope checks for customer and product facts.
- Read-only HTTP/MCP integration patterns; no customer messaging or ERP write client.
- SQLite request idempotency, per-customer serialization, cancellation, and restart recovery.

The built-in ERP SQL adapter is an example that must be reviewed and adapted to your own schema. Do not connect it to production until access controls and queries have been independently reviewed. Never place real customer data, credentials, or provider responses in public issues or test fixtures.

## Install and run

Requires Python 3.11 or later.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
# Set real values in your shell or private JSON config; .env.example is reference only.
export BRAIN_API_TOKEN='replace-with-a-long-random-secret'
export OPENAI_API_KEY='replace-with-your-provider-key'
uvicorn sales_brain_pydantic.app:app --host 127.0.0.1 --port 8000
```

The example values are placeholders; replace them before running model-backed requests. The application reads process environment variables and an optional JSON file; it does not automatically load `.env`. Keep secrets outside the repository. `BRAIN_CONFIG_FILE` defaults to `./service.json`; the service requires a regular file with restrictive permissions when that file exists. Environment variables override JSON values.

## Offline tests

The test suite uses synthetic fixtures and local ASGI/loopback services. It does not need a model key, business dataset, ERP, or external service.

```sh
python -m pip install -e '.[test]'
pytest -q
```

## API overview

- `GET /health` returns non-secret service readiness information.
- `GET /v1/status` requires `Authorization: Bearer <BRAIN_API_TOKEN>`.
- `POST /v1/chat` requires the same bearer token and returns a proposed response.
- `GET /v1/runs/{run_id}` and `POST /v1/runs/{run_id}/cancel` inspect or cancel a run.
- `PUT /v1/customers/{customer_id}/takeover` controls a local manual-takeover guard.

See [docs/integrations.md](docs/integrations.md) for adapter boundaries and [SECURITY.md](SECURITY.md) for security guidance.
