import hashlib, json
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class Config:
    # retrieval
    corpus: str = "MedRAG/textbooks"
    query_encoder: str = "ncbi/MedCPT-Query-Encoder"
    article_encoder: str = "ncbi/MedCPT-Article-Encoder"
    # MedCPT's query encoder truncates here and drops the rest silently.
    # Shared by the retriever (guards + truncates) and the rewriter
    # (enforces its output stays under it) — hence config, not either module.
    max_query_tokens: int = 64
    cache_dir: str = ".cache/corpus"
    cache_embed: str = ".cache/embeddings"

    # search
    article_collection: str = "textbook_corpus"
    top_k: int = 5
    use_hybrid: bool = False
    rrf_k: int = 60

    # generation
    # Which SDK rag/llm.py reaches for. The two fields move together —
    # switching provider without switching model id is an immediate 404.
    llm_provider: str = "gemini"    # "openai" | "gemini" | "anthropic"
    # Pinned, not "gemini-flash-latest": a floating alias silently changes the
    # model under a results table. 3.7-flash returns 503 under load — the
    # newest model is not the safest one to run 300 sequential eval calls on.
    llm_model: str = "gemini-3.5-flash"
    temperature: float = 0.0
    max_context_chunks: int = 5

    # agent
    use_rewriter: bool = True
    use_agent_loop: bool = True
    max_clarifying_questions: int = 4

    def hash(self) -> str:
        """Short stable hash — used in eval result filenames."""
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()[:8]
