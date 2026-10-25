"""ADK callback for saving a generated blog post as Markdown."""

import os
from pathlib import Path

from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from .markdown_publisher import MarkdownPublisher


def publish_generated_article(
    callback_context: CallbackContext,
) -> types.Content:
    """Save the writer's completed output and return a concise confirmation."""
    markdown = callback_context.state.get("blog_writer_result")
    if not isinstance(markdown, str) or not markdown.strip():
        raise RuntimeError("The blog writer produced no Markdown article.")

    output_directory = Path(os.environ.get("BLOG_OUTPUT_DIR", "output"))
    output_path = MarkdownPublisher(output_directory).publish(markdown)
    callback_context.state["final_blog_post_path"] = str(output_path)
    return types.Content(
        role="model",
        parts=[types.Part(text=f"Blog post saved to {output_path}")],
    )
