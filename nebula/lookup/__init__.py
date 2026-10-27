"""nebula/lookup: Nebula's built-in MCP servers for the Lookup star (off until the owner turns them on).

    docs.py   search the folders the owner adds (plain text, Markdown, code, JSON, HTML), read a file from them
    web.py    search the web (Brave Search with a key from the environment, else DuckDuckGo's HTML page) and read a page

Each is a small MCP server over stdio (newline-delimited JSON-RPC 2.0), started by the hub like any other server.
"""
