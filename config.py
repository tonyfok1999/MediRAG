import hashlib, json
from dataclasses import dataclass, asdict

from dotenv import load_dotenv

# Every entry point imports Config, so .env loads exactly once, here —
# rather than in whichever script happened to remember to call it.
load_dotenv()

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
    llm_model: str = "..."          # your API model
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
