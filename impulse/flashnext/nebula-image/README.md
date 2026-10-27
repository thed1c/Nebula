# Impulse Flash-Next: the Nebula image

`Containerfile` makes the image Nebula runs Impulse Flash-Next from: the reference engine's image plus Impulse in `/opt/impulse/bin`
(`impulse-fx`, `libimpulse_fx.so`, `libstrata_kq.so`, `impulse-server`), the layout the reference engine runs since One Engine P3
(2026-10-25, `nebula-flashnext:impulse-36f62a1-r768`, which also carries its run config at
`/opt/flashnext/run-config-fx.json`).

`nebula/flashnext.py` builds the engine definition: a container `command` engine whose command is

    /opt/impulse/bin/impulse-fx -m /models/<pack dir> --host 127.0.0.1 --port <port>
        --fx-image-python /opt/strata-serve/.venv/bin/python --fx-config-json <run config JSON>

with the run config Nebula builds passed inline, or `--fx-config <path>` for a config the image carries (the reference engine's:
`engine_spec(..., config_path="/opt/flashnext/run-config-fx.json")`). The run config is Strata's (server.py's) without
`exe`; its `"log"` (`/sandbox/engine.log`) gets the engine's stderr, appended, as server.py's did (impulse-fx after
P3, branch one-engine-p3-fix; an image with an older impulse-fx writes the engine's lines to the container output only).

The earlier layout (`start.py` writing `IMPULSE_FLASHNEXT_CONFIG` to a file and starting `serve.server`) is gone.
