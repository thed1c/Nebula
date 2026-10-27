# The decision plan (nebula-plan-v1)

What chooses to do what, when: Nebula's **Plan** tab shows it as one row per job, a job's class and the models it may
use, in order, and lets you edit it. The module is `nebula/plan.py`.

```json
{"schema": "nebula-plan-v1", "name": "home",
 "classes":  [{"name": "interactive", "rank": 0, "tolerance_s": 60}, {"name": "agent", "rank": 1, "tolerance_s": 15}],
 "models":   [{"id": "big", "name": "Big model", "slots": 1}, {"id": "small", "name": "Small model", "slots": 4}],
 "jobs":     [{"id": "studio", "name": "The Studio", "class": "interactive", "order": ["big", "small"],
               "ticked": true, "tags": ["service.studio/*"]}],
 "settings": {"park_wait_s": 600, "sticky_s": 7200}}
```

The rules every row shares:
- A call takes the first model in its job's order with a free slot.
- When none is free, it waits up to its class's `tolerance_s` for its first choice, then takes the next model that
  can hold it.
- A call too big for all of them is parked for up to `park_wait_s`.
- A run stays on its first model for `sticky_s`.

`ticked: false` means the order is the job's default (nothing chosen for it yet). `tags` are the caller labels (globs)
a job covers.

**This Nebula's own plan** comes from its config: a job per agent (its ticks are its models) and one for requests
that name no agent (the default ticks). **Apply** writes it back: agents' `models`, `default_models`,
`rules.tolerance_seconds` and `rules.parking_wait_seconds`.

**A watched Switchyard's plan** (the optional watch) is its `/switchyard/plan`, shown beside this Nebula's own, read
only. Nebula never changes a Switchyard it watches.
