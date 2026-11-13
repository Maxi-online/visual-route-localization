# Private configuration

Keep API keys, bot tokens, proxy credentials, private keys, production settings, exports and personal datasets outside Git. Commit empty examples and synthetic fixtures only. Use environment variables or a secret manager for live services.

The offline demo makes no provider calls. Full applications may access paid APIs, external sources, messaging integrations and databases when configured. Use your own isolated deployment and review the selected entrypoint.

Secret scan checks current files and incoming commits with pinned Gitleaks and redacted output. It is a pattern-based check, not proof that every possible secret is absent. Rotate an exposed credential in its owning provider; ordinary file removal does not invalidate it.

Do not paste credentials or real user records into issues, logs or screenshots. Share a minimal synthetic reproduction. Keep upstream attribution and licenses intact.
