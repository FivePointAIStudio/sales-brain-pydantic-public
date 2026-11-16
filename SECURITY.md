# Security policy

This project handles conversation text, customer identifiers, provider credentials, integration responses, and local audit state when configured by an integrator. Treat these as sensitive.

## Reporting a vulnerability

Do not publish credentials, customer data, private endpoints, or exploit details in a public issue. Contact the maintainers through the private contact channel provided by the repository host, with a concise impact description and reproduction steps using synthetic data.

## Deployment guidance

- Keep the API bound to loopback or place it behind an authenticated, access-controlled reverse proxy.
- Generate a unique high-entropy `BRAIN_API_TOKEN`; never use the example placeholder.
- Store provider credentials and private registries outside source control with restrictive file permissions.
- Configure integrations for read-only access and review each endpoint, identity mapping, and query for your environment.
- Protect the SQLite state directory according to your data policy; it may contain conversation content and audit records.
- Apply network egress controls. The application permits configured local HTTP integrations by default; provider HTTPS access is controlled separately by the model SDK.
- Pin and review dependencies before deployment. Local tests do not constitute a security audit or production acceptance.
