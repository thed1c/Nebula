# Syslog

Optional, and off by default. Turn it on in the Syslog panel on Captain's Log, or in the config:

```json
"syslog": {"enabled": true, "host": "192.0.2.10", "port": 514, "protocol": "udp", "level": "warning",
           "security": true, "facility": "local0"}
```

Nebula then sends its own warnings and errors (what the Error log shows) and, when `"security"` is on, the security
log's events to your syslog server, one RFC 5424 line each:

```
<132>1 2026-11-04T13:21:05.655Z myhost nebula 41234 log - nebula.teamwork: a work item failed: ...
<132>1 2026-11-04T13:21:07.201Z myhost nebula 41234 security - security: signin.refused who=192.0.2.7
```

| Field | |
|---|---|
| `host`, `port` | the syslog server (514 for UDP and TCP, 6514 for TLS) |
| `protocol` | `udp` (one datagram a line), `tcp` (octet-counted, RFC 6587) or `tls` (RFC 5425, the server's certificate checked against this computer's) |
| `level` | `warning` (warnings and errors) or `error` (errors only) |
| `security` | the security log too: each event's kind, who and what, never its details ([security-log.md](security-log.md)) |
| `facility` | `user`, `daemon`, `auth`, `authpriv` or `local0` to `local7` |

The MSGID says where a line came from: `log` for Nebula's own log, `security` for the security log, `test` for the
panel's **Send a test line**. A message is one line (newlines become spaces), cut at 2,048 bytes.

**What is in it.** Nebula's log lines name models, engines, work items and what went wrong; Nebula does not log
prompts or replies, but an engine's error is passed on in the engine's own words. UDP and TCP are plain text on the
wire: use TLS for a server across a network you do not trust.

**When the server is down.** Lines go out from one background thread, so a server that is slow or down never holds
up a request: a line that cannot be sent is counted as failed (the panel shows the last error), and if more than
2,000 wait, the newest are dropped and counted. Turning it on or off, or pointing it somewhere else, is written to the
security log (`syslog.settings`).
