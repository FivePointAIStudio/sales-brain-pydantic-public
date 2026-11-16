# Implementation notes

This repository is a standalone reference implementation. It contains an evidence-aware request model, a PydanticAI agent, a FastAPI application, a local SQLite state store, and adapter patterns for read-only service and MCP integrations.

The service does not send messages or execute business writes. Local action receipts represent proposals only. Model output is untrusted: trusted identity and region must come from configured server-side adapters, and the model cannot select arbitrary endpoints, credentials, or SQL statements.

Before connecting an integration, map your own identity model, document source ownership and freshness, validate read-only scopes, and review the adapter's fixed query templates against your schema. No integration or business result is certified by this repository.
