# Security

## Secrets

Aurum reads secrets **only from environment variables**. YAML configs never hold them, and
the config loader rejects credential-like keys.

| Variable | Used by |
|---|---|
| `ANTHROPIC_API_KEY` | LLM trading desk (`aurum desk run/replay`, `live.use_desk`) |
| `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`, `MT5_PATH` | MetaTrader 5 broker adapter (Windows) |
| `AURUM_ALERT_WEBHOOK_URL`, `AURUM_ALERT_TELEGRAM_CHAT_ID` | optional alert sinks |
| `AURUM_ARTIFACT_KEY` | optional HMAC signing of pickled live artifacts |

Copy `.env.example` to `.env` (git-ignored), fill it in, never commit it, and export it into
your shell before running Aurum, for example `set -a; . ./.env; set +a`. Aurum does not read
`.env` files itself. Credentials are never logged, and journals and decision logs truncate
tool payloads.

## ⚠️ Historic credential exposure (v1)

Early v1 commits (`222f813`, `3a1fb32`, `a528c1b`, `a11d227`, `2f5f331`) contained a
MetaAPI token and account ID, and they are still in the public git history. **That token
must be revoked in the MetaAPI dashboard.** Treat it as compromised whatever the current
code contains. Rewriting public history (`git filter-repo`) is optional once the token is
revoked, because a revoked token is harmless.

## Pickled artifacts

Live artifacts contain pickled Python objects. **Load only artifacts you produced
yourself.** Unpickling untrusted files can execute arbitrary code. Set `AURUM_ARTIFACT_KEY`
to have artifacts HMAC-signed on save and verified on load.

## Trading safety controls

- `aurum live run` is **dry-run by default**.
- A non-demo account is refused unless `live.allow_live_real: true` is set **and**
  `--i-understand-real-money` is passed.
- The risk manager's kill switch (daily loss, maximum drawdown) persists across restarts
  and needs a human `reset_halt(confirm="RESET")`.
- The bot only touches positions carrying its own magic number.
- The LLM desk can never bypass the risk manager. Its output is a bounded forecast that goes
  through the same sizer and risk checks.

## Reporting

Please report vulnerabilities privately through GitHub's "Report a vulnerability" rather than
in a public issue.
