def _tags_sentence(tags: list[str]) -> str:
    return " ".join(t.replace("_", " ").strip() for t in tags if t.strip())
