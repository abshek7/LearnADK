"""Root assistant with one web researcher and one Markdown blog writer."""

import os

from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool

from .blog_research.prompts import load_prompt
from .blog_research.publishing_callback import publish_generated_article
from .blog_research.tavily_tool import tavily_search
from .groq_model import GroqModel

DEFAULT_GROQ_FALLBACK_MODELS = (
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
)
configured_fallback_models = tuple(
    model.strip()
    for model in os.environ.get("GROQ_FALLBACK_MODELS", "").split(",")
    if model.strip()
)
MODEL = GroqModel(
    model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
    fallback_models=configured_fallback_models or DEFAULT_GROQ_FALLBACK_MODELS,
)

web_searcher = LlmAgent(
    model=MODEL,
    name="web_searcher",
    description=(
        "Searches the web with Tavily for an in-scope research request and "
        "returns concise, source-linked research notes."
    ),
    instruction=load_prompt("researcher.txt"),
    tools=[tavily_search],
    output_key="research_notes",
)

blog_writer = LlmAgent(
    model=MODEL,
    name="blog_writer",
    description=(
        "Turns web research into an evidence-based Markdown blog post and "
        "saves the post to the configured output directory."
    ),
    instruction=load_prompt("writer.txt"),
    output_key="blog_writer_result",
    after_agent_callback=publish_generated_article,
)

root_agent = LlmAgent(
    model=MODEL,
    name="blog_assistant",
    description=(
        "A focused web-research and blog-writing assistant. It can answer "
        "brief in-scope questions, research a topic, and publish a sourced "
        "Markdown blog post."
    ),
    instruction=load_prompt("root.txt"),
    tools=[AgentTool(web_searcher), AgentTool(blog_writer)],
)
