# Router status v1

Nebula can watch another Switchyard by its URL, read only (optional): its world is drawn on Nebula's page (its models,
its agents, its callers, and its calls in flight as lights between them). Nebula polls `<url>/switchyard/status` with
`GET` and writes nothing back. This page is the shape that answer has; every Nebula serves it, and any router can.
The schema is [router-status-v1.schema.json](router-status-v1.schema.json), and `nebula/examples/fake_router.py`
serves a working example.

```json
"watch": {"url": "http://127.0.0.1:18300", "poll_seconds": 2}
```

The answer is a JSON object; an optional top-level `name` is what the page calls the Switchyard (else its address). Fields not
listed here are ignored, so a router can carry more. Only `capacity.destinations` and `in_flight_calls` are needed;
every other field may be left out, and the page then shows less.

## The models: `capacity.destinations`

An object, one entry per model (a destination calls can go to), keyed by its name.

| Field | Type | Meaning |
|---|---|---|
| `label` | string | display name (the key when missing) |
| `routes` | list of strings | the route ids that lead to this model; a call names a route |
| `slots` | integer | how many calls it takes at once |
| `busy` | integer | calls running on it now |
| `queued` | integer | calls waiting for it |
| `slot_context` | integer | context length per slot, in tokens |
| `avg_call_seconds` | number | the average call's length |
| `unavailable` | string, boolean or null | why it is out of use; null or missing when it is in use |

## How a model is reached: `routes`

An object keyed by route id.

| Field | Type | Meaning |
|---|---|---|
| `base_url` | string | the engine's address, shown on the page |
| `ready` | boolean | whether this route answers; a model is up when any of its routes is ready (or it lists none) and it is not `unavailable` |

## The agents: `switchyard.actors`

An object: actor name → mode (a short string such as `"live"` or `"off"`, shown as given). The watch's own
`agents` list (optional, in Nebula's config) picks which of them to show and in what order, matched without case.

## The calls: `in_flight_calls`

A list of the calls running now. Calls are seen only while they are in flight, so one shorter than a poll can be
missed.

| Field | Type | Meaning |
|---|---|---|
| `actor` | string | the agent that made it |
| `caller` | string | the caller tag (e.g. `writer-app/draft`), named by the callers below |
| `run` | string | the run or session it belongs to |
| `route` | string | the route it went to; its model is the destination that lists the route |
| `since` | number | when it started, Unix epoch seconds |

`actor`, `caller`, `run` and `since` together identify a call between polls.

## Waiting and parked: `waiting`, `capacity.parking`

`waiting` lists requests waiting for a slot: `id`, `caller`, `actor`, `class`, `group` (the model it waits for),
`reason`, `since`, `label`. `capacity.parking` lists requests no model can take now; each entry is an object, passed
to the page as it is.

## Other destinations: `external_destinations.destinations`

A list of destinations that are not engines (a CLI, a webhook): `id` (required), `kind`, `enabled` (only enabled
ones are shown), `status` (`credential_missing` or `missing` shows it as down), `running`, `queued`.

## The callers

A caller is `{"id", "name", "class", "tags": [glob, ...]}`. The first caller whose glob matches a call's caller tag
names its source on the page; with no callers, the tag itself is the source. Nebula takes them from the first of:

1. a `callers` list in the status itself;
2. `<url>/switchyard/registry` (`{"callers": [...]}`), read at most every 30 seconds.

Nebula reaches a watched Switchyard only through its HTTP interface, never through files on disk.

## The plan (optional)

A Switchyard that wants its decision plan shown on the Plan tab serves it as JSON in the shape of
[plan-v1.md](plan-v1.md) at `<url>/switchyard/plan`. It is shown beside this Nebula's own, read only: Nebula never
sends a plan back.

## Lanes and decisions (optional)

These sections feed the **Lanes** and **Decisions** widgets. Each one is optional, and without it those widgets
show less. All are read-only, like everything here.

**`admission.groups`**: an object, one entry per lane (normally the same names as `capacity.destinations`): `label`,
`size` (slots), `busy`, `free`, `queued`, `on_hold` (why the lane is held, or null), and `holders`, a list of
`{actor, route, since}`, one per slot in use.

**`admission.stats`**: `waits`, `timeouts`, `longest_seconds`, and `last`, the latest admission: `id`, `actor`,
`caller`, `class`, `group` (the lane), `label`, `route`, `outcome` (e.g. `admitted`, `timeout`), `reason`, `seconds`
(how long it waited), `behind` (the actors it waited behind), `since`, `ended_at`.

**`route_policy`**: `queues` (per lane: `routes`, `slots`, `timeout_seconds`, `max_queued` per class), `routes` (per
route: `enabled`, `priority`, `group`, `state` such as `qualified` or `retired`, `model`, `configured_context`, and
`eligible`, an object of capability → boolean: vision, tools, thinking, …) and `counters` (any named counts, e.g.
redirects, refused_ineligible, queue_full).

**`routes.*`** may also carry `state`, `reason` (why it is not ready), `since`, `model_file`,
`expected_model_file` (shown as a warning when they differ) and `attest`.

**`capacity.destinations.*`** may also carry `free`, `measured` and `queued_by_class` (class → count).

**`capacity.mode`, `capacity.counters`, `capacity.last`**: the capacity router. `counters` holds `decisions` (mode →
count), `agree` and `differ` (a shadow policy's pick against the live one) and `parked`. `last` is the latest
decision: `at`, `kind`, `mode`, `caller`, `actor`, `class`, `candidates`, `rank`, `pick`, `selected`, `actual`,
`agree`, `reason`, `wait_s`, `est_tokens`.

**`switchyard.mode`, `switchyard.decisions`, `switchyard.errors`, `switchyard.last`**: the routing itself.
`decisions` is actor/tier → count. `last` holds `at`, `actor`, `caller`, `mode`, `tier`, `switchyard_order` (the
routes in the order tried), `selected_route`, `routed`, `applied`, `differs`, `upstream_failure` and `ms`.

A router reports only its **last** decision of each kind. Nebula keeps the feed itself: each poll, a `last` it has not
seen yet is added (the newest 200 are kept), so a decision that comes and goes between two polls is missed, like a
short call.

