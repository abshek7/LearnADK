"""Load agent instructions kept separately under the prompts directory."""

from pathlib import Path

_PROMPT_DIRECTORY = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> str:
    """Read a named UTF-8 prompt file and surface missing-file errors."""
    return (_PROMPT_DIRECTORY / name).read_text(encoding="utf-8")
