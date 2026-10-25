"""Markdown post output with safe, readable filenames."""

import re
from pathlib import Path


class MarkdownPublisher:
    """Persist generated Markdown in a configured output directory."""

    def __init__(self, output_directory: Path) -> None:
        self.output_directory = output_directory

    def publish(self, markdown: str) -> Path:
        """Write a Markdown article named from its first H1 heading."""
        if not markdown.strip():
            raise ValueError("Cannot publish an empty Markdown article.")

        title = self._title(markdown)
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80]
        filename = f"{slug or 'blog-post'}.md"
        self.output_directory.mkdir(parents=True, exist_ok=True)
        output_path = self.output_directory / filename
        output_path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
        return output_path.resolve()

    @staticmethod
    def _title(markdown: str) -> str:
        for line in markdown.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return "blog-post"
