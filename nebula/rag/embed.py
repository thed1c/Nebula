"""nebula/rag/embed.py: local embeddings and a local reranker. Nothing leaves this machine.

- **fastembed** (ONNX on the CPU, `pip install "nebula-eh[lookup]"`): `BAAI/bge-small-en-v1.5` by default (384
  dimensions), and the reranker `Xenova/ms-marco-MiniLM-L-6-v2`. Their files are fetched from Hugging Face once, on
  first use, into Nebula's data folder (`nebula-lookup/models`), then read from there.
- **An engine of this Nebula's own** (`"lookup": {"embed": {"tier": "<a tier>"}}`): its `/v1/embeddings` (llama.cpp
  started with `--embedding`, say). A cloud tier is refused: a document's text never goes to a provider.
"""
from __future__ import annotations

import json
import os
import threading

DEFAULT_EMBED = "BAAI/bge-small-en-v1.5"
DEFAULT_RERANK = "Xenova/ms-marco-MiniLM-L-6-v2"


class EmbedError(RuntimeError):
    pass


def fastembed_available() -> bool:
    try:
        import fastembed  # noqa: F401
        return True
    except ImportError:
        return False


class FastEmbedder:
    def __init__(self, model: str = DEFAULT_EMBED, cache_dir: str | None = None):
        self.model_name, self.cache_dir = model, cache_dir or os.environ.get("FASTEMBED_CACHE_PATH")
        self._m = None
        self.lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"fastembed:{self.model_name}"

    def _model(self):
        if self._m is None:
            from fastembed import TextEmbedding
            self._m = TextEmbedding(self.model_name, cache_dir=self.cache_dir)
        return self._m

    def embed(self, texts: list[str]) -> list[list[float]]:
        with self.lock:
            return [list(map(float, v)) for v in self._model().embed(texts, batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        with self.lock:
            m = self._model()
            q = getattr(m, "query_embed", None)
            return list(map(float, next(iter(q([text]) if q else m.embed([text])))))

    def dims(self) -> int:
        return len(self.embed_query("dimensions"))


class EngineEmbedder:
    """A local engine's /v1/embeddings (OpenAI shape)."""

    def __init__(self, engine, tier: str, model: str | None = None):
        self.engine, self.tier, self.model = engine, tier, model
        self._dims = None

    @property
    def name(self) -> str:
        return f"engine:{self.tier}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        body = {"input": texts, **({"model": self.model} if self.model else {})}
        try:
            r = self.engine._open("/v1/embeddings", body)
            d = json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            raise EmbedError(f"the engine {self.tier} gave no embeddings: {e}") from None
        rows = sorted(d.get("data") or [], key=lambda x: x.get("index", 0))
        return [list(map(float, x["embedding"])) for x in rows]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def dims(self) -> int:
        if self._dims is None:
            self._dims = len(self.embed_query("dimensions"))
        return self._dims


class Reranker:
    def __init__(self, model: str = DEFAULT_RERANK, cache_dir: str | None = None):
        self.model_name, self.cache_dir = model, cache_dir or os.environ.get("FASTEMBED_CACHE_PATH")
        self._m = None
        self.lock = threading.Lock()

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        with self.lock:
            if self._m is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                self._m = TextCrossEncoder(self.model_name, cache_dir=self.cache_dir)
            return [float(s) for s in self._m.rerank(query, texts)]
