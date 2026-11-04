"""Pick the model backend: Claude via the Anthropic API, or a local open-source model."""
import config

_cached = None


def make_client():
    """RAG_PROVIDER=anthropic (default, needs ANTHROPIC_API_KEY) or RAG_PROVIDER=local (no key)."""
    global _cached
    if _cached is None:
        if config.PROVIDER == "local":
            from local_llm import LocalClient
            _cached = LocalClient()
        else:
            import anthropic
            _cached = anthropic.Anthropic()
    return _cached


def model_label() -> str:
    if config.PROVIDER == "local":
        import os
        from local_llm import DEFAULT_FILE
        return os.path.basename(os.getenv("RAG_LOCAL_MODEL_PATH") or os.getenv("RAG_LOCAL_FILE", DEFAULT_FILE))
    return config.MODEL
