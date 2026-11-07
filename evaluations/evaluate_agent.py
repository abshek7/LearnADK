"""Evaluate the live blog assistant with DeepEval."""

import asyncio
from datetime import datetime, timezone
import importlib
import json
import logging
import os
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

logger = logging.getLogger(__name__)

OPENROUTER_CHAT_COMPLETIONS_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)
DEFAULT_JUDGE_MODEL = "nvidia/nemotron-3.5-super-vl:free"
DEFAULT_JUDGE_FALLBACK_MODELS = (
    "xingchen-agi/xing4.0-29b-a4b:free",
)
MAX_RATE_LIMIT_RETRIES = 2
RATE_LIMIT_RETRY_SECONDS = 2.0

EVALUATION_PROMPT = (
    "Research practical uses of retrieval-augmented generation in customer "
    "support. Write a concise, sourced article that covers benefits and "
    "limitations."
)
EXPECTED_ARTICLE_DESCRIPTION = (
    "A concise Markdown article about practical uses of retrieval-augmented "
    "generation in customer support, including evidence-based benefits, "
    "limitations, and citations to the sources used for research."
)
RESEARCHER_TOOL_NAME = "web_searcher"
WRITER_TOOL_NAME = "blog_writer"
DEFAULT_REPORT_DIRECTORY = "evaluation_reports"
DEFAULT_MAX_COMPLETION_TOKENS = 8192
METRIC_DESCRIPTIONS = {
    "AnswerRelevancyMetric": (
        "Breaks the article into statements and judges whether each statement "
        "answers the user's request. The score is the relevant-statement ratio."
    ),
    "FaithfulnessMetric": (
        "Extracts claims from the article and facts from the research notes, "
        "then judges whether each claim is supported by those facts. The score "
        "is the supported-claim ratio."
    ),
}


class OpenRouterEvaluationModel(DeepEvalBaseLLM):
    """Use OpenRouter's free model variants as DeepEval's judge."""

    def __init__(self) -> None:
        self.api_key = os.environ["OPENROUTER_API_KEY"]
        self.model_id = os.environ.get("DEEPEVAL_MODEL", DEFAULT_JUDGE_MODEL)
        configured_fallbacks = os.environ.get(
            "DEEPEVAL_FALLBACK_MODELS",
            ",".join(DEFAULT_JUDGE_FALLBACK_MODELS),
        )
        self.fallback_models = tuple(
            model.strip()
            for model in configured_fallbacks.split(",")
            if model.strip() and model.strip() != self.model_id
        )
        self._validate_free_models((self.model_id, *self.fallback_models))
        self.max_completion_tokens = int(
            os.environ.get(
                "DEEPEVAL_MAX_COMPLETION_TOKENS",
                str(DEFAULT_MAX_COMPLETION_TOKENS),
            )
        )
        if self.max_completion_tokens < 1:
            raise ValueError("DEEPEVAL_MAX_COMPLETION_TOKENS must be greater than 0.")
        self.model_attempts: list[str] = []
        self.successful_models: list[str] = []
        self.structured_output_errors: list[dict[str, str]] = []
        self.http_client = httpx.Client(
            headers=self._headers(),
            timeout=httpx.Timeout(120.0),
        )
        super().__init__(model=self.model_id)

    @staticmethod
    def _validate_free_models(model_ids: tuple[str, ...]) -> None:
        """Prevent configuration from selecting paid, Gemini, or Groq models."""
        for model_id in model_ids:
            if not model_id.endswith(":free"):
                raise ValueError(
                    f"Judge model {model_id!r} must use an OpenRouter :free variant."
                )
            lowered_model_id = model_id.lower()
            if "gemini" in lowered_model_id:
                raise ValueError("Gemini models are not allowed as judge models.")
            if "groq" in lowered_model_id:
                raise ValueError("Groq models are not allowed as judge models.")

    def load_model(self) -> httpx.Client:
        return self.http_client

    def get_model_name(self) -> str:
        return self.model_id

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "LearnADK DeepEval",
        }

    @staticmethod
    def _response_format(options: dict[str, Any]) -> dict[str, Any]:
        """Ask the judge for JSON matching DeepEval's schema when available."""
        request_options = dict(options)
        schema = request_options.pop("schema", None)
        if schema is not None:
            request_options["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__.lower(),
                    "strict": False,
                    "schema": schema.model_json_schema(),
                },
            }
        return request_options

    def _request_body(
        self,
        model_id: str,
        prompt: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.max_completion_tokens,
            "provider": {"require_parameters": True},
            **options,
        }

    @staticmethod
    def _retry_delay(error: httpx.HTTPStatusError | None) -> float:
        if error is None:
            return RATE_LIMIT_RETRY_SECONDS
        retry_after = error.response.headers.get("Retry-After")
        try:
            return (
                max(float(retry_after), 0.0)
                if retry_after
                else RATE_LIMIT_RETRY_SECONDS
            )
        except ValueError:
            return RATE_LIMIT_RETRY_SECONDS

    @staticmethod
    def _response_text(
        response: httpx.Response,
        model_id: str,
        schema: Any = None,
    ) -> str:
        response.raise_for_status()
        response_data = response.json()
        choices = response_data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(
                f"OpenRouter judge {model_id!r} returned no choices."
            )
        choice = choices[0]
        content = choice.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                f"OpenRouter judge {model_id!r} returned an empty response."
            )
        finish_reason = choice.get("finish_reason")
        if finish_reason in {"length", "max_tokens"}:
            raise ValueError(
                f"OpenRouter judge {model_id!r} truncated its response "
                f"(finish_reason={finish_reason!r})."
            )
        if schema is not None:
            normalized_content = content.strip()
            if (
                normalized_content.startswith("```")
                and normalized_content.endswith("```")
            ):
                normalized_content = re.sub(
                    r"^```(?:json)?\s*|\s*```$",
                    "",
                    normalized_content,
                    flags=re.IGNORECASE,
                )
            try:
                validated = schema.model_validate_json(normalized_content)
            except (ValueError, TypeError) as error:
                validation_errors = getattr(error, "errors", None)
                if callable(validation_errors):
                    error_details = validation_errors(include_input=False)
                    if isinstance(error_details, list):
                        summaries = []
                        for item in error_details:
                            location = ".".join(map(str, item["loc"]))
                            summaries.append(
                                f"{location}: {item['msg']}"
                                if location
                                else item["msg"]
                            )
                        error_summary = "; ".join(summaries)
                    else:
                        error_summary = ""
                else:
                    error_summary = str(error)
                raise ValueError(
                    f"OpenRouter judge {model_id!r} returned invalid "
                    f"{schema.__name__} JSON: {error_summary or error}"
                ) from error
            return validated.model_dump_json()
        return content

    def _record_structured_output_error(
        self, model_id: str, error: ValueError
    ) -> None:
        self.structured_output_errors.append(
            {"model": model_id, "error": str(error)}
        )
        logger.warning(
            "OpenRouter free judge %s returned invalid structured output: %s",
            model_id,
            error,
        )

    def generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        schema = kwargs.get("schema")
        request_options = self._response_format(kwargs)
        candidate_models = (self.model_id, *self.fallback_models)
        last_rate_limit: httpx.HTTPStatusError | None = None
        last_output_error: ValueError | None = None
        last_output_error_model: str | None = None

        for retry in range(MAX_RATE_LIMIT_RETRIES + 1):
            last_rate_limit = None
            for model_id in candidate_models:
                try:
                    self.model_attempts.append(model_id)
                    response = self.http_client.post(
                        OPENROUTER_CHAT_COMPLETIONS_URL,
                        json=self._request_body(model_id, prompt, request_options),
                    )
                    content = self._response_text(response, model_id, schema)
                    self.successful_models.append(model_id)
                    return content
                except httpx.HTTPStatusError as error:
                    if error.response.status_code != 429:
                        raise
                    last_rate_limit = error
                    logger.warning(
                        "OpenRouter free judge %s is rate-limited.", model_id
                    )
                except ValueError as error:
                    last_output_error = error
                    last_output_error_model = model_id
                    self._record_structured_output_error(model_id, error)

            if last_rate_limit is not None and retry < MAX_RATE_LIMIT_RETRIES:
                delay = self._retry_delay(last_rate_limit)
                logger.warning(
                    "All configured OpenRouter free judge models are "
                    "rate-limited; retrying in %.2f seconds.",
                    delay,
                )
                time.sleep(delay)

            if retry == MAX_RATE_LIMIT_RETRIES:
                if last_output_error is not None:
                    raise RuntimeError(
                        "All configured OpenRouter free judge models failed to "
                        f"return valid structured output; last failure from "
                        f"{last_output_error_model}: {last_output_error}"
                    ) from last_output_error
                if last_rate_limit is None:
                    raise RuntimeError(
                        "OpenRouter evaluation model attempts ended unexpectedly."
                    )

        if last_output_error is not None:
            raise RuntimeError(
                "OpenRouter free judge models failed to return valid structured "
                f"output; last failure from {last_output_error_model}: "
                f"{last_output_error}"
            ) from last_output_error
        if last_rate_limit is not None:
            raise last_rate_limit
        raise RuntimeError("OpenRouter evaluation model attempts ended unexpectedly.")

    async def a_generate(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        schema = kwargs.get("schema")
        request_options = self._response_format(kwargs)
        candidate_models = (self.model_id, *self.fallback_models)
        last_rate_limit: httpx.HTTPStatusError | None = None
        last_output_error: ValueError | None = None
        last_output_error_model: str | None = None

        async with httpx.AsyncClient(
            headers=self._headers(),
            timeout=httpx.Timeout(120.0),
        ) as client:
            for retry in range(MAX_RATE_LIMIT_RETRIES + 1):
                last_rate_limit = None
                for model_id in candidate_models:
                    try:
                        self.model_attempts.append(model_id)
                        response = await client.post(
                            OPENROUTER_CHAT_COMPLETIONS_URL,
                            json=self._request_body(
                                model_id,
                                prompt,
                                request_options,
                            ),
                        )
                        content = self._response_text(response, model_id, schema)
                        self.successful_models.append(model_id)
                        return content
                    except httpx.HTTPStatusError as error:
                        if error.response.status_code != 429:
                            raise
                        last_rate_limit = error
                        logger.warning(
                            "OpenRouter free judge %s is rate-limited.", model_id
                        )
                    except ValueError as error:
                        last_output_error = error
                        last_output_error_model = model_id
                        self._record_structured_output_error(model_id, error)

                if last_rate_limit is not None and retry < MAX_RATE_LIMIT_RETRIES:
                    delay = self._retry_delay(last_rate_limit)
                    logger.warning(
                        "All configured OpenRouter free judge models are "
                        "rate-limited; retrying in %.2f seconds.",
                        delay,
                    )
                    await asyncio.sleep(delay)

                if retry == MAX_RATE_LIMIT_RETRIES:
                    if last_output_error is not None:
                        raise RuntimeError(
                            "All configured OpenRouter free judge models failed "
                            "to return valid structured output; last failure "
                            f"from {last_output_error_model}: {last_output_error}"
                        ) from last_output_error
                    if last_rate_limit is None:
                        raise RuntimeError(
                            "OpenRouter evaluation model attempts ended unexpectedly."
                        )

        if last_output_error is not None:
            raise RuntimeError(
                "OpenRouter free judge models failed to return valid structured "
                f"output; last failure from {last_output_error_model}: "
                f"{last_output_error}"
            ) from last_output_error
        if last_rate_limit is not None:
            raise last_rate_limit
        raise RuntimeError("OpenRouter evaluation model attempts ended unexpectedly.")


def _function_calls_from_event(event: Any) -> list[Any]:
    """Return tool calls recorded in one ADK event."""
    get_function_calls = getattr(event, "get_function_calls", None)
    if callable(get_function_calls):
        return get_function_calls()

    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    return [
        part.function_call
        for part in parts
        if getattr(part, "function_call", None) is not None
    ]


def _function_responses_from_event(event: Any) -> list[Any]:
    """Return tool responses recorded in one ADK event."""
    get_function_responses = getattr(event, "get_function_responses", None)
    if callable(get_function_responses):
        return get_function_responses()

    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    return [
        part.function_response
        for part in parts
        if getattr(part, "function_response", None) is not None
    ]


def _text_from_tool_response(tool_response: Any) -> str:
    """Extract successful text from an ADK tool response."""
    response_data = getattr(tool_response, "response", None)
    if isinstance(response_data, dict):
        error_message = response_data.get("error")
        if error_message:
            tool_name = getattr(tool_response, "name", "<unknown>")
            raise AssertionError(f"Agent tool {tool_name} failed: {error_message}")
        result = response_data.get("result")
    else:
        result = response_data

    if not isinstance(result, str) or not result.strip():
        tool_name = getattr(tool_response, "name", "<unknown>")
        raise AssertionError(f"Agent tool {tool_name} returned no text.")
    return result


def _get_research_notes_from_events(
    events: list[Any],
) -> tuple[str, list[str]]:
    """Check specialist tool order and return research notes and call names."""
    tool_calls = [
        tool_call
        for event in events
        for tool_call in _function_calls_from_event(event)
    ]
    called_tool_names = [getattr(tool_call, "name", None) for tool_call in tool_calls]

    try:
        researcher_call_index = called_tool_names.index(RESEARCHER_TOOL_NAME)
        writer_call_index = called_tool_names.index(WRITER_TOOL_NAME)
    except ValueError as error:
        raise AssertionError(
            "Expected calls to both specialist tools; observed tool calls: "
            f"{called_tool_names}"
        ) from error

    if researcher_call_index >= writer_call_index:
        raise AssertionError("The blog writer ran before the web researcher.")

    tool_responses = [
        response
        for event in events
        for response in _function_responses_from_event(event)
    ]
    responses_by_tool_name = {
        getattr(response, "name", None): response for response in tool_responses
    }

    for tool_name in (RESEARCHER_TOOL_NAME, WRITER_TOOL_NAME):
        if tool_name not in responses_by_tool_name:
            raise AssertionError(f"The {tool_name} tool call has no response.")

    _text_from_tool_response(responses_by_tool_name[WRITER_TOOL_NAME])
    research_notes = _text_from_tool_response(
        responses_by_tool_name[RESEARCHER_TOOL_NAME]
    )
    return research_notes, called_tool_names


def _extract_urls(text: str) -> set[str]:
    """Find plain-text HTTP(S) URLs and trim punctuation after each URL."""
    return {
        url.rstrip(".,;:")
        for url in re.findall(r"https?://[^\s)>\]]+", text)
    }


def _to_json_compatible(value: Any) -> Any:
    """Convert metric details such as Pydantic verdicts to report-safe values."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _validate_required_api_keys() -> None:
    """Stop early with a clear message when live-service keys are missing."""
    required_keys = ("GROQ_API_KEY", "TAVILY_API_KEY", "OPENROUTER_API_KEY")
    missing_keys = [key for key in required_keys if not os.environ.get(key)]
    if missing_keys:
        raise SystemExit(
            "Set the following environment variables before running the "
            f"evaluation: {', '.join(missing_keys)}"
        )


def _load_agent_and_runner() -> tuple[Any, Any]:
    """Import the project agent and ADK runner after environment setup."""
    agent_module = importlib.import_module(f"{PROJECT_ROOT.name}.agent")
    runner_class = importlib.import_module("google.adk.runners").InMemoryRunner
    return agent_module.root_agent, runner_class


def _run_agent(agent: Any, runner_class: Any, output_directory: str) -> list[Any]:
    """Run the article request, directing the published file to a temp folder."""
    previous_output_directory = os.environ.get("BLOG_OUTPUT_DIR")
    os.environ["BLOG_OUTPUT_DIR"] = output_directory

    try:
        runner = runner_class(agent=agent, app_name="blog_research_deepeval")
        return asyncio.run(
            runner.run_debug(
                EVALUATION_PROMPT,
                user_id="deepeval",
                session_id=str(uuid.uuid4()),
                quiet=True,
            )
        )
    finally:
        if previous_output_directory is None:
            os.environ.pop("BLOG_OUTPUT_DIR", None)
        else:
            os.environ["BLOG_OUTPUT_DIR"] = previous_output_directory


def _read_published_article(output_directory: str) -> tuple[Path, str]:
    """Return the one Markdown article written by the agent."""
    article_paths = list(Path(output_directory).glob("*.md"))
    if len(article_paths) != 1:
        raise AssertionError(
            "Expected exactly one published Markdown article; "
            f"found {len(article_paths)}."
        )

    article_path = article_paths[0]
    return article_path, article_path.read_text(encoding="utf-8")


def _check_article_sources(article: str, research_notes: str) -> dict[str, Any]:
    """Compare article citations with URLs found in the research notes."""
    if not research_notes.strip():
        raise AssertionError("The researcher returned no text to ground the article.")

    article_urls = sorted(_extract_urls(article))
    research_urls = sorted(_extract_urls(research_notes))
    unsupported_urls = sorted(set(article_urls) - set(research_urls))
    return {
        "passed": bool(article_urls) and not unsupported_urls,
        "article_urls": article_urls,
        "research_urls": research_urls,
        "unsupported_urls": unsupported_urls,
    }


def _evaluate_article(
    article: str,
    research_notes: str,
) -> tuple[list[dict[str, Any]], OpenRouterEvaluationModel]:
    """Run DeepEval metrics and collect scores, explanations, and verdicts."""
    test_case = LLMTestCase(
        input=EVALUATION_PROMPT,
        actual_output=article,
        expected_output=EXPECTED_ARTICLE_DESCRIPTION,
        retrieval_context=[research_notes],
    )
    judge_model = OpenRouterEvaluationModel()
    metrics = (
        AnswerRelevancyMetric(threshold=0.7, model=judge_model, async_mode=False),
        FaithfulnessMetric(threshold=0.7, model=judge_model, async_mode=False),
    )
    metric_reports = []
    for metric in metrics:
        metric_name = metric.__class__.__name__
        metric_description = METRIC_DESCRIPTIONS.get(
            metric_name, "DeepEval LLM-as-a-judge metric."
        )
        try:
            metric.measure(test_case, _show_indicator=False)
            metric_passed = bool(metric.success)
            metric_reports.append(
                {
                    "name": metric_name,
                    "description": metric_description,
                    "score": metric.score,
                    "threshold": metric.threshold,
                    "passed": metric_passed,
                    "status": "PASS" if metric_passed else "FAIL",
                    "reason": metric.reason,
                    "evaluation_model": metric.evaluation_model,
                    "details": _to_json_compatible({
                        field: getattr(metric, field)
                        for field in ("statements", "truths", "claims", "verdicts")
                        if hasattr(metric, field)
                    }),
                    "verbose_logs": metric.verbose_logs,
                }
            )
        except Exception as error:
            metric_reports.append(
                {
                    "name": metric_name,
                    "description": metric_description,
                    "score": metric.score,
                    "threshold": metric.threshold,
                    "passed": False,
                    "status": "ERROR",
                    "reason": metric.reason,
                    "evaluation_model": metric.evaluation_model,
                    "details": _to_json_compatible({
                        field: getattr(metric, field)
                        for field in ("statements", "truths", "claims", "verdicts")
                        if hasattr(metric, field)
                    }),
                    "error": f"{type(error).__name__}: {error}",
                    "verbose_logs": metric.verbose_logs,
                }
            )
    return metric_reports, judge_model


def _report_directory() -> Path:
    """Resolve the configured report folder relative to the project root."""
    configured_directory = Path(
        os.environ.get("DEEPEVAL_REPORT_DIR", DEFAULT_REPORT_DIRECTORY)
    )
    if not configured_directory.is_absolute():
        configured_directory = PROJECT_ROOT / configured_directory
    configured_directory.mkdir(parents=True, exist_ok=True)
    return configured_directory


def _write_evaluation_report(report: dict[str, Any]) -> tuple[Path, Path]:
    """Save both machine-readable JSON and readable Markdown reports."""
    report_directory = _report_directory()
    report_stem = f"evaluation-{report['run_id']}"
    json_path = report_directory / f"{report_stem}.json"
    markdown_path = report_directory / f"{report_stem}.md"

    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_format_markdown_report(report), encoding="utf-8")
    return markdown_path, json_path


def _format_markdown_report(report: dict[str, Any]) -> str:
    """Format the evaluation details as a report that is easy to review."""
    citation_check = report["citation_check"]
    lines = [
        "# Blog assistant evaluation",
        "",
        f"- **Run:** `{report['run_id']}`",
        f"- **Started:** {report['started_at']}",
        f"- **Overall result:** {report['status']}",
        f"- **Article file:** `{report['article_file']}`",
        f"- **Evaluation method:** {report['evaluation_method']}",
        f"- **Configured judge:** {report['judge']['configured_model']}",
        f"- **Fallback judges:** {', '.join(report['judge']['fallback_models']) or 'none'}",
        f"- **Judge completion-token limit:** {report['judge']['max_completion_tokens']}",
        f"- **Judge model attempts:** {', '.join(report['judge']['model_attempts']) or 'none'}",
        f"- **Successful judge models:** {', '.join(report['judge']['successful_models']) or 'none'}",
        "",
        "## What was evaluated",
        "",
        "1. The root agent called the researcher before the writer.",
        "2. The writer returned a successful response and published one Markdown article.",
        "3. Every article URL appeared in the research notes.",
        "4. DeepEval answer relevancy and faithfulness scores met their thresholds.",
        "",
        "## Workflow and citation checks",
        "",
        f"- **Tool calls:** {' -> '.join(report['workflow']['tool_calls'])}",
        f"- **Research then writing:** {'PASS' if report['workflow']['passed'] else 'FAIL'}",
        f"- **Citation validation:** {'PASS' if citation_check['passed'] else 'FAIL'}",
        f"- **Article URLs:** {', '.join(citation_check['article_urls']) or 'none'}",
        f"- **Research URLs:** {', '.join(citation_check['research_urls']) or 'none'}",
    ]
    if report["judge"]["structured_output_errors"]:
        lines.extend(
            [
                "",
                "### Judge structured-output errors",
                "",
            ]
        )
        lines.extend(
            f"- **{failure['model']}:** {failure['error']}"
            for failure in report["judge"]["structured_output_errors"]
        )

    if citation_check["unsupported_urls"]:
        lines.append(
            "- **Unsupported URLs:** "
            + ", ".join(citation_check["unsupported_urls"])
        )

    lines.extend(["", "## DeepEval metric results", ""])
    for metric in report["metrics"]:
        score = metric["score"]
        score_text = "not produced" if score is None else f"{score:.3f}"
        lines.extend(
            [
                f"### {metric['name']}: {metric['status']}",
                "",
                metric["description"],
                "",
                f"- **Score:** {score_text}",
                f"- **Required threshold:** {metric['threshold']:.2f}",
                f"- **Judge model:** {metric['evaluation_model'] or 'unknown'}",
                f"- **Reason:** {metric['reason'] or 'No reason returned.'}",
            ]
        )
        if metric.get("error"):
            lines.append(f"- **Error:** `{metric['error']}`")
        if metric["details"]:
            lines.extend(
                [
                    "",
                    "```json",
                    json.dumps(metric["details"], indent=2, ensure_ascii=False, default=str),
                    "```",
                ]
            )
        if metric.get("verbose_logs"):
            lines.extend(["", "<details>", "<summary>DeepEval verbose log</summary>", ""])
            lines.extend(["```text", metric["verbose_logs"], "```", "", "</details>"])
        lines.append("")

    lines.extend(
        [
            "## Evaluation input",
            "",
            report["input"]["prompt"],
            "",
            "### Expected output description",
            "",
            report["input"]["expected_output"],
            "",
            "## Research context used for faithfulness",
            "",
            "```text",
            report["research_notes"],
            "```",
            "",
            "## Article evaluated",
            "",
            report["article"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run the agent, evaluate its output, and persist a detailed report."""
    load_dotenv(PROJECT_ROOT / ".env")
    _validate_required_api_keys()
    run_started_at = datetime.now(timezone.utc)
    run_id = run_started_at.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    agent, runner_class = _load_agent_and_runner()

    with tempfile.TemporaryDirectory(prefix="blog-research-eval-") as output_directory:
        events = _run_agent(agent, runner_class, output_directory)
        research_notes, tool_call_names = _get_research_notes_from_events(events)
        article_path, article = _read_published_article(output_directory)
        citation_check = _check_article_sources(article, research_notes)
        metric_reports, judge_model = _evaluate_article(
            article,
            research_notes,
        )

        started_at = run_started_at.isoformat()
        workflow_check = {
            "passed": (
                tool_call_names.index(RESEARCHER_TOOL_NAME)
                < tool_call_names.index(WRITER_TOOL_NAME)
            ),
            "tool_calls": tool_call_names,
            "article_published": True,
        }
        report = {
            "run_id": run_id,
            "started_at": started_at,
            "status": (
                "ERROR"
                if any(metric["status"] == "ERROR" for metric in metric_reports)
                else (
                    "PASS"
                    if workflow_check["passed"]
                    and citation_check["passed"]
                    and all(metric["passed"] for metric in metric_reports)
                    else "FAIL"
                )
            ),
            "passed": (
                workflow_check["passed"]
                and citation_check["passed"]
                and all(metric["passed"] for metric in metric_reports)
            ),
            "article_file": article_path.name,
            "input": {
                "prompt": EVALUATION_PROMPT,
                "expected_output": EXPECTED_ARTICLE_DESCRIPTION,
            },
            "workflow": workflow_check,
            "citation_check": citation_check,
            "evaluation_method": (
                "The workflow and URL checks are deterministic application checks. "
                "The agent still uses Groq; DeepEval uses the configured OpenRouter "
                "free-tier judge to score "
                "answer relevancy and faithfulness on a 0-to-1 scale; each score must meet its "
                "0.70 threshold. Judge calls allow up to "
                f"{judge_model.max_completion_tokens} completion tokens. "
                "Structured judge responses are schema-validated; malformed or "
                "truncated responses trigger another configured free judge."
            ),
            "judge": {
                "configured_model": judge_model.model_id,
                "fallback_models": list(judge_model.fallback_models),
                "max_completion_tokens": judge_model.max_completion_tokens,
                "model_attempts": judge_model.model_attempts,
                "successful_models": judge_model.successful_models,
                "structured_output_errors": judge_model.structured_output_errors,
            },
            "metrics": metric_reports,
            "research_notes": research_notes,
            "article": article,
        }
        markdown_report_path, json_report_path = _write_evaluation_report(report)

        print(f"Evaluation result: {report['status']}")
        for metric in metric_reports:
            score = metric["score"]
            score_text = "not produced" if score is None else f"{score:.3f}"
            print(
                f"{metric['name']}: {score_text} "
                f"(threshold {metric['threshold']:.2f}) - "
                f"{metric['status']}"
            )
        print(f"Readable report: {markdown_report_path}")
        print(f"Full JSON report: {json_report_path}")

        if not report["passed"]:
            raise SystemExit(
                "Evaluation did not pass. Review the saved report for metric "
                "scores, explanations, and any errors."
            )


if __name__ == "__main__":
    main()
