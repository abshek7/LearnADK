# Web Research Blog Assistant

A Python assistant built with [Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/) that researches topics on the web and turns source-backed findings into Markdown blog posts.

The application has three agents:

1. **`blog_assistant` (root)** — handles greetings and in-scope questions, declines unrelated requests, and coordinates research requests.
2. **`web_searcher`** — searches the web with Tavily and returns concise research notes with source titles and URLs.
3. **`blog_writer`** — uses the research notes to write an evidence-based Markdown article. After the writer finishes, an ADK callback saves the response to disk.

The flow for an article is **root → web searcher → blog writer → Markdown file**. There are no separate planner, parallel researcher, or critic agents.

## Features

- Web search through Tavily.
- Groq chat-completions model support through a small ADK adapter.
- Configurable Groq model fallbacks and rate-limit retries.
- Separate prompt files for the root, researcher, and writer.
- Markdown output named from the article's first H1 heading.
- Focused unit tests for search, model failover, publishing, and the agent prompts.

## Requirements

- Python 3.10 or newer.
- A [Groq API key](https://console.groq.com/keys).
- A [Tavily API key](https://app.tavily.com/).

## Setup

Create and activate a virtual environment, then install the dependencies.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and replace the placeholders with your API keys. ADK loads the `.env` file for the agent when starting it. Keep `.env` private; it is excluded by `.gitignore`. Never commit API keys.

## Configuration

| Variable | Required | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | Yes | API key used for model requests. |
| `GROQ_MODEL` | No | Primary Groq chat model. The sample `.env.example` selects `llama-3.3-70b-versatile`; the code fallback is `openai/gpt-oss-120b`. |
| `GROQ_FALLBACK_MODELS` | No | Comma-separated model IDs available to your Groq account. When omitted, the code uses `llama-3.3-70b-versatile` and `openai/gpt-oss-20b` as fallbacks. |
| `TAVILY_API_KEY` | Yes for web research | API key used by Tavily Search. |
| `BLOG_OUTPUT_DIR` | No | Directory for generated Markdown files. Defaults to `output`. Relative paths are resolved from the process working directory. |

The active search provider is Tavily. The Google Search integration files and the `GOOGLE_API_KEY` / `GOOGLE_SEARCH_MODEL` sample settings are retained in the repository, but are not used by the current agent workflow.

### Groq rate limits

When Groq returns a rate-limit error, the adapter tries the configured models. If every model in the list is rate-limited, it waits for the retry delay provided by Groq and makes up to two more passes through the list. This can help with short bursts, but does not increase account quota or guarantee a request will succeed if the models remain rate-limited.

## Run the ADK web interface

From the project root, start the local development UI:

```bash
adk web .
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), select the application, and ask for a research-based article. For example:

> Research WebMCP and explain its implementation and use cases for enterprises and startups.

The root agent searches the topic, passes the returned research notes to the writer, and saves the resulting article. The assistant reports the saved file path. By default, generated posts are written under `output/`.

The ADK development UI is intended for local development. Do not expose it to an untrusted or public network without adding appropriate authentication and deployment controls.

## Run the tests

From the project root with the virtual environment active:

```bash
python -m unittest discover -s tests -v
```

The Tavily client and Groq model behavior are tested with mocked responses; running the unit tests does not require live API calls.

## Project layout

```text
.
├── agent.py                         # Root agent and two specialist agents
├── blog_research/
│   ├── google_search.py              # Retained Google ADK search integration
│   ├── markdown_publisher.py         # Safe Markdown filenames and file output
│   ├── prompts.py                    # Prompt loader
│   ├── publishing_callback.py        # Saves the writer's completed response
│   ├── tavily_client.py              # Tavily Search API client
│   └── tavily_tool.py                # Tavily tool used by the web searcher
├── prompts/
│   ├── researcher.txt
│   ├── root.txt
│   └── writer.txt
├── tests/
│   └── test_blog_research.py
├── .env.example
└── requirements.txt
```

## Scope and limitations

- Prompt instructions guide scope and source handling; they are not a hard security boundary.
- Research quality depends on the search results returned by Tavily. The agents should not treat search excerpts as proof beyond what those excerpts support.
- Groq model availability, context limits, and quotas depend on the account and selected models.
- Markdown output is saved locally; this project does not publish posts to a hosted CMS.
