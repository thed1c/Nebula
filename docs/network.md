# Model machines on your network

One Nebula, many machines: another machine's cards become this Nebula's, the way this computer's are. Three models
on one box look the same as one model on each of three boxes, to you and to anyone you share with. No Nebula is
installed on a model machine, and nothing between the machines travels in plain HTTP.

- **Add a server (SSH)** (Models › Add a model): its host key is shown first and must be confirmed (and is checked on
  every later connection: a changed key is refused). Nebula then looks at the machine (OS, GPUs, Python, Podman, its
  firewall) and lists what needs an administrator, ticked: install Podman (rootless), keep your services running while
  nobody is logged in, and let the two machines through each firewall. Your sudo password for those is asked in the
  page, given to sudo once, and never kept. Then it copies the model-machine pieces over SSH and starts them as your
  own services there:
  - the network's node programs (NVIDIA's, Apache-2.0), which carry this Nebula's requests to that machine over mutual TLS;
  - the node keeper (`nebula/nodekeeper.py`, one file): it keeps those running, and passes each request to the
    engine serving its model on that machine;
  - NVIDIA OpenShell on rootless Podman, where this Nebula starts its engines on that machine's cards.

  The machine joins your network by a PIN this Nebula carries itself, and its OpenShell gateway is registered here,
  managed over SSH. Your SSH agent or a key file signs in; no password is stored.
- **Its models** are routed like your own: a request goes from this computer's router to that machine's over mTLS,
  and the node keeper hands it to the engine. Who may use what is decided here, before anything leaves this computer.
- **Never plain HTTP between machines:** a model at another machine's `http://` address is refused (Add a model and
  every call).
- **What it needs:** Linux with systemd (user services), python3, NVIDIA's driver and container toolkit for its
  cards, and sudo once for Podman (apt for now). Ports between the two machines: TCP 14318-14323 and 8080, UDP 5353.
- **Removing it** stops and deletes the engines' sandboxes there and everything Nebula put there; Podman stays.
