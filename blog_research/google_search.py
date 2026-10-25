"""Create Google ADK search-grounding tools for the Groq-led workflow."""

from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.tools.google_search_agent_tool import GoogleSearchAgentTool


def create_google_search_tool(model: str) -> GoogleSearchAgentTool:
    """Wrap ADK's Gemini-grounded Google Search agent as a tool."""
    search_agent = LlmAgent(
        model=model,
        name="google_search_agent",
        description="Searches Google and returns a grounded response with sources.",
        instruction=(
            "Use Google Search to research the user's query. Summarize relevant "
            "findings and include source titles and URLs from the search "
            "grounding. Distinguish uncertainty and do not invent facts or "
            "citations."
        ),
        tools=[google_search],
    )
    return GoogleSearchAgentTool(agent=search_agent)
