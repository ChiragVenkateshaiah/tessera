"""sentence-transformers implementation of the Embedder interface."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from tessera.embedding.base import Embedder

# Small (~90MB), CPU-friendly, no API key. Good default for a local pilot;
# swapped for Bedrock Titan/Cohere in Phase 4 behind this same interface.
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbedder(Embedder):
    """Local embedding model via sentence-transformers.

    embed_query and embed_documents do the same thing for this particular
    model (it has no query/passage distinction) — the split still exists on
    the interface (see Embedder) so a model that does need it is a drop-in
    swap, not an interface change.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(list(texts), show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        embedding = self._model.encode([text], show_progress_bar=False)
        return embedding[0].tolist()
