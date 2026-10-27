# OpenTelemetry

Optional, and off by default. When you turn it on (System tab, or `"otel": {"enabled": true, "endpoint": ...}`), Nebula
sends one span per turn to your collector over OTLP/HTTP (JSON). Grafana, Jaeger, Honeycomb and the like read them.

## What a span holds: metadata only

| Attribute | |
|---|---|
| `gen_ai.operation.name`, `gen_ai.system` | `chat`, `nebula` |
| `gen_ai.request.model`, `gen_ai.response.model` | the model asked for, the one that answered |
| `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `nebula.cached_tokens` | the counts |
| `gen_ai.response.finish_reasons`, `nebula.completeness` | how it ended |
| `nebula.input`, `nebula.tier`, `nebula.engine` | which door it came in by, the model's name here, its engine's address |
| `nebula.tenant`, `nebula.guest`, `nebula.caller`, `nebula.agent`, `nebula.priority` | who it was for, and its class at the gates |
| `nebula.queued_s`, `nebula.tool_rounds`, `nebula.retries` | the wait at the gates, the tool rounds, the retries before the first byte |

**Never sent:** a prompt, a reply, a system prompt, a tool's arguments or result, a key, or an error's text (only its
status). The tests send a prompt holding a password and check that neither it nor the reply reaches the collector.

## Where it goes

The collector's address must be `https://`, or `http://` on this machine (a collector running beside Nebula), so
metadata never crosses the network in the clear. `"headers"` carries a collector's API key when it needs one. Spans
are batched and sent from one background thread. A collector that is down loses spans, never a turn; the System tab
shows what was sent and what failed. Turning it on or off, or changing the address, is written to the security log.
