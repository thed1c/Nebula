"""Nebula (in the code, "the cockpit"): the server above a local engine such as Strata-Lite (docs/DESIGN.md).

Step 1, the gateway: OpenAI Chat Completions, Anthropic Messages and OpenAI Responses in; the engine's own
/v1/chat/completions out. Nothing here imports from strata/ or lite/; the engine is an address.
"""
__version__ = "0.1.4"
