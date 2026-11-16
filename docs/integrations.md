# Integration guide

The service accepts private configuration through `BRAIN_CONFIG_FILE` and optional environment overrides. See `.env.example` for safe placeholder values. Service and MCP registries are supplied by the integrator and must never be copied from a live environment into this repository.

## Adapter expectations

- HTTP integrations use fixed service names and routes, bearer credentials from private configuration, bounded request and response sizes, timeouts, and no redirects.
- MCP endpoints must be explicitly enabled and resolve to loopback HTTP addresses. Model-supplied arguments cannot choose an endpoint or credential.
- Customer identity and region are resolved from a trusted server-side profile. An ambiguous or unconfirmed identity must not grant access to customer-specific data.
- Catalog, pricing, stock, orders, fulfillment, and other facts need source timestamps and evidence references. Knowledge-model answers are reference material, not authoritative live facts.
- Example SQL templates in `src/sales_brain_pydantic/integrations.py` must be reviewed and adapted to the integrator's schema. Keep them read-only and bind customer scope from trusted identity, never model text.

The secondary knowledge provider allows HTTP only at loopback addresses and `/v1`, plus the RFC 5737 documentation-only address `192.0.2.10` for non-routable examples; it does not permit arbitrary remote hosts such as `example.com`. Use a local development stub. No real remote endpoint is included.

Do not use sample/test credentials or synthetic identities outside local tests. Protect logs and SQLite state as potentially sensitive.
