# Changelog

## 0.1.3

**One engine for every model.** Impulse now runs every lane: the Flash-Next packs and the GGUF models start from the same
engine, so there is one program to install and update.

**Long context, if you want it (off by default).** A Flash-Next model can now read up to 512K or 1M tokens of context
instead of 256K. Turn it on per model under **Long context: off / 512K / 1M**. Up to 256K nothing changes at all; past
it the model stretches its position encoding as the conversation grows (dynamic YaRN). Each request pays for it with a
somewhat smaller expert cache, so answers can come a little slower and differ slightly from the 256K setting. In our
tests a model at 512K found a passkey hidden in a 300,000-token document and wrote at about the same speed.

**More than one request at once, if you want it (off by default).** A model can answer several people at the same time
(**parallel** per model) instead of one after another. The second person no longer waits for the first; total speed
stays about the same or drops a little, depending on the model and the cards. A request that runs alone keeps its full
speed, including speculative decoding (MTP); requests that share the engine decode one token at a time. With long context
on a two-card setup, Nebula picks the engine mode that supports it by itself. Answers are consistent but not always
identical, character for character, to the same request run alone.

**Updates only when you ask.** Nebula can look for a new version and tell you on the page ("Update ready"); it is
opt-in, there is no daily timer, and the default is "Only when I ask". Update feeds are signed.

**Fixes**
- After a restart, Nebula takes back a model it was already running in a sandbox on another machine instead of showing it
  as stopped.
- Halo's router sends its own key on every call, so Halo answers only Nebula.

**Experimental (not tested on real hardware yet)**
- Spreading one model's experts over two machines over the network (TCP, with encryption and authentication) and an
  RDMA transport for faster links. They are built and tested on one machine only; leave them off unless you want to try
  them.
