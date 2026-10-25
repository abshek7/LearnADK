"""Focused tests for web-result parsing and Markdown publishing."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

import httpx
from groq import RateLimitError
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.tools.google_search_tool import GoogleSearchTool

from blog_research.google_search import create_google_search_tool
from blog_research.markdown_publisher import MarkdownPublisher
from blog_research.publishing_callback import publish_generated_article
from blog_research.tavily_client import TavilySearchClient
from blog_research.tavily_tool import tavily_search
from groq_model import GroqModel


class GoogleSearchToolTests(unittest.TestCase):
    def test_wraps_adk_google_search_with_the_configured_gemini_model(self) -> None:
        tool = create_google_search_tool("gemini-3.8-flash")

        self.assertEqual(tool.name, "google_search_agent")
        self.assertTrue(tool.propagate_grounding_metadata)
        self.assertEqual(tool.agent.model, "gemini-3.8-flash")
        self.assertEqual(len(tool.agent.tools), 1)
        self.assertIsInstance(tool.agent.tools[0], GoogleSearchTool)


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class TavilySearchClientTests(unittest.TestCase):
    def test_search_sends_query_and_normalizes_results(self) -> None:
        response_body = json.dumps(
            {
                "results": [
                    {
                        "title": "Climate report",
                        "url": "https://example.com/report",
                        "content": "An excerpt",
                        "score": 0.95,
                        "published_date": "2026-10-01",
                    }
                ]
            }
        ).encode("utf-8")
        with patch(
            "blog_research.tavily_client.urlopen",
            return_value=_FakeResponse(response_body),
        ) as urlopen_mock:
            results = TavilySearchClient(api_key="test-key").search(
                "climate evidence", limit=1
            )

        request = urlopen_mock.call_args.args[0]
        self.assertEqual(request.full_url, TavilySearchClient.endpoint)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(
            json.loads(request.data),
            {
                "query": "climate evidence",
                "search_depth": "basic",
                "max_results": 1,
                "include_answer": False,
                "include_raw_content": False,
            },
        )
        self.assertEqual(results[0].title, "Climate report")
        self.assertEqual(results[0].url, "https://example.com/report")
        self.assertEqual(results[0].content, "An excerpt")
        self.assertEqual(results[0].score, 0.95)

    def test_requires_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "TAVILY_API_KEY"):
                TavilySearchClient().search("a query")

    def test_rejects_empty_query(self) -> None:
        with self.assertRaises(ValueError):
            TavilySearchClient(api_key="test-key").search(" ")


class TavilyToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_searches_with_the_supplied_query(self) -> None:
        with patch(
            "blog_research.tavily_tool._SEARCH_CLIENT.search",
            return_value=[],
        ) as search:
            result = await tavily_search("El Niño impacts")

        search.assert_called_once_with("El Niño impacts", 5)
        self.assertEqual(result["query"], "El Niño impacts")

    async def test_rejects_empty_query(self) -> None:
        with self.assertRaises(ValueError):
            await tavily_search(" ")


class GroqModelTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _rate_limit_error(
        retry_after: str | None = "0.25",
        message: str = "rate limited",
    ) -> RateLimitError:
        headers = {"retry-after": retry_after} if retry_after else {}
        response = httpx.Response(
            429,
            headers=headers,
            request=httpx.Request("POST", "https://api.groq.com"),
        )
        return RateLimitError(
            message, response=response, body={"error": "rate limit"}
        )

    async def test_switches_models_after_a_rate_limit(self) -> None:
        rate_limit_error = self._rate_limit_error()
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="response", tool_calls=[])
                )
            ],
            model="fallback-model",
        )
        create = AsyncMock(side_effect=[rate_limit_error, completion])
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )
        model = GroqModel(
            model="primary-model",
            fallback_models=("fallback-model",),
        )
        request = cast(
            LlmRequest,
            SimpleNamespace(
                config=SimpleNamespace(system_instruction=None, tools=[]),
                contents=[],
            ),
        )

        with patch.object(model, "_get_client", return_value=client):
            responses = [
                item
                async for item in model.generate_content_async(request)
            ]

        self.assertEqual(
            [call.kwargs["model"] for call in create.await_args_list],
            ["primary-model", "fallback-model"],
        )
        self.assertEqual(responses[0].model_version, "fallback-model")

    async def test_waits_and_retries_all_models_when_all_are_rate_limited(self) -> None:
        completion = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="response", tool_calls=[])
                )
            ],
            model="primary-model",
        )
        create = AsyncMock(
            side_effect=[
                self._rate_limit_error("0.5"),
                self._rate_limit_error("0.5"),
                completion,
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )
        model = GroqModel(
            model="primary-model",
            fallback_models=("fallback-model",),
        )
        request = cast(
            LlmRequest,
            SimpleNamespace(
                config=SimpleNamespace(system_instruction=None, tools=[]),
                contents=[],
            ),
        )

        with (
            patch.object(model, "_get_client", return_value=client),
            patch("groq_model.asyncio.sleep", new_callable=AsyncMock) as sleep,
        ):
            responses = [
                item
                async for item in model.generate_content_async(request)
            ]

        self.assertEqual(
            [call.kwargs["model"] for call in create.await_args_list],
            ["primary-model", "fallback-model", "primary-model"],
        )
        sleep.assert_awaited_once_with(0.5)
        self.assertEqual(responses[0].model_version, "primary-model")

    async def test_parses_millisecond_retry_delay_in_groq_message(self) -> None:
        error = self._rate_limit_error(
            retry_after=None,
            message="Rate limit reached. Please try again in 352.5ms.",
        )

        from groq_model import _rate_limit_retry_delay

        self.assertAlmostEqual(_rate_limit_retry_delay(error), 0.3525)


class AgentPromptTests(unittest.TestCase):
    def test_prompts_describe_the_three_agent_pipeline(self) -> None:
        from blog_research.prompts import load_prompt

        root_prompt = load_prompt("root.txt")
        researcher_prompt = load_prompt("researcher.txt")
        writer_prompt = load_prompt("writer.txt")

        self.assertIn("web_searcher", root_prompt)
        self.assertIn("blog_writer", root_prompt)
        self.assertIn("tavily_search exactly once", researcher_prompt)
        self.assertIn("saved as a Markdown file", writer_prompt)
        self.assertIn("does not need to make a publishing tool", root_prompt)
        self.assertIn("{research_notes}", writer_prompt)


class PublishingCallbackTests(unittest.TestCase):
    def test_publishes_writer_output_and_returns_the_path(self) -> None:
        markdown = "# Research Findings\n\nSupported article."
        context = SimpleNamespace(
            state={"blog_writer_result": markdown}
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.dict(
                os.environ, {"BLOG_OUTPUT_DIR": temporary_directory}
            ):
                result = publish_generated_article(cast(CallbackContext, context))

            output_text = result.parts[0].text if result.parts else None
            if not isinstance(output_text, str):
                self.fail("Publishing callback did not return confirmation text.")
            output_path = Path(output_text.removeprefix("Blog post saved to "))
            self.assertTrue(output_path.is_file())
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                markdown + "\n",
            )
            self.assertEqual(
                context.state["final_blog_post_path"],
                str(output_path),
            )


class MarkdownPublisherTests(unittest.TestCase):
    def test_publishes_with_a_safe_title_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            publisher = MarkdownPublisher(Path(temporary_directory))
            output_path = publisher.publish(
                "# Web Research: A Practical Guide\n\nArticle text."
            )

            self.assertEqual(output_path.name, "web-research-a-practical-guide.md")
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "# Web Research: A Practical Guide\n\nArticle text.\n",
            )

    def test_rejects_empty_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            publisher = MarkdownPublisher(Path(temporary_directory))
            with self.assertRaises(ValueError):
                publisher.publish("  ")


if __name__ == "__main__":
    unittest.main()
