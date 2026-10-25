"""Direct Groq chat-completions adapter for Google ADK."""

import asyncio
import json
import logging
import os
import re
from collections.abc import AsyncGenerator
from typing import Any

from groq import AsyncGroq, RateLimitError
from pydantic import PrivateAttr

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types


logger = logging.getLogger(__name__)
MAX_RATE_LIMIT_RETRIES = 2


class GroqModel(BaseLlm):
    """Use Groq's chat completions API without a provider translation layer."""

    fallback_models: tuple[str, ...] = ()
    _client: AsyncGroq | None = PrivateAttr(default=None)

    def _get_client(self) -> AsyncGroq:
        if self._client is None:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Add it to the agent's .env file "
                    "or set it in the environment before starting ADK."
                )
            self._client = AsyncGroq(api_key=api_key, max_retries=0)
        return self._client

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Generate one complete turn; token streaming is not enabled."""
        del stream
        messages = _messages_from_request(llm_request)
        tools = _tools_from_request(llm_request)

        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"

        candidates = tuple(dict.fromkeys((self.model, *self.fallback_models)))
        client = self._get_client()
        last_rate_limit: RateLimitError | None = None
        for retry in range(MAX_RATE_LIMIT_RETRIES + 1):
            for index, model in enumerate(candidates):
                try:
                    completion = await client.chat.completions.create(
                        **{**request, "model": model}
                    )
                    break
                except RateLimitError as error:
                    last_rate_limit = error
                    if index < len(candidates) - 1:
                        logger.warning(
                            "Groq model %s is rate-limited; trying fallback "
                            "model %s.",
                            model,
                            candidates[index + 1],
                        )
            else:
                if retry == MAX_RATE_LIMIT_RETRIES:
                    if last_rate_limit is None:
                        raise RuntimeError("Groq model attempts ended unexpectedly.")
                    raise last_rate_limit

                delay = _rate_limit_retry_delay(last_rate_limit)
                logger.warning(
                    "All configured Groq models are rate-limited; retrying "
                    "the model list in %.2f seconds.",
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            break

        choice = completion.choices[0]
        response_parts: list[types.Part] = []

        if choice.message.content:
            response_parts.append(types.Part(text=choice.message.content))

        for tool_call in choice.message.tool_calls or []:
            arguments = json.loads(tool_call.function.arguments or "{}")
            response_parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        args=arguments,
                    )
                )
            )

        yield LlmResponse(
            content=types.Content(role="model", parts=response_parts),
            model_version=completion.model,
            partial=False,
            turn_complete=True,
        )


def _messages_from_request(llm_request: LlmRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system_instruction = _text_from_value(llm_request.config.system_instruction)
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})

    for content in llm_request.contents:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_responses: list[dict[str, str]] = []

        for part in content.parts or []:
            if part.text is not None:
                text_parts.append(part.text)
            elif part.function_call is not None:
                function_call = part.function_call
                if not function_call.id:
                    raise ValueError("ADK function calls must include a call ID.")
                tool_calls.append(
                    {
                        "id": function_call.id,
                        "type": "function",
                        "function": {
                            "name": function_call.name,
                            "arguments": json.dumps(function_call.args or {}),
                        },
                    }
                )
            elif part.function_response is not None:
                function_response = part.function_response
                if not function_response.id:
                    raise ValueError(
                        "ADK function responses must include the originating "
                        "call ID."
                    )
                tool_responses.append(
                    {
                        "role": "tool",
                        "tool_call_id": function_response.id,
                        "name": function_response.name or "",
                        "content": json.dumps(function_response.response or {}),
                    }
                )
            elif part.inline_data is not None or part.file_data is not None:
                raise ValueError(
                    "The direct Groq adapter supports text and function calls, "
                    "not image or file inputs."
                )

        if tool_responses:
            messages.extend(tool_responses)
            if text_parts:
                messages.append(
                    {
                        "role": _chat_role(content.role),
                        "content": "".join(text_parts),
                    }
                )
        elif tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": "".join(text_parts) or None,
                    "tool_calls": tool_calls,
                }
            )
        elif text_parts:
            messages.append(
                {
                    "role": _chat_role(content.role),
                    "content": "".join(text_parts),
                }
            )

    return messages


def _chat_role(role: str | None) -> str:
    if role == "model":
        return "assistant"
    if role == "user":
        return "user"
    raise ValueError(f"Unsupported ADK conversation role: {role!r}")


def _text_from_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, types.Part):
        return value.text or ""
    if isinstance(value, types.Content):
        return "".join(part.text or "" for part in value.parts or [])
    if isinstance(value, list):
        return "".join(_text_from_value(item) for item in value)
    return ""


def _tools_from_request(llm_request: LlmRequest) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for tool in llm_request.config.tools or []:
        for declaration in tool.function_declarations or []:
            parameters = declaration.parameters_json_schema
            if parameters is None and declaration.parameters is not None:
                parameters = declaration.parameters.model_dump(
                    mode="json", by_alias=True, exclude_none=True
                )
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": declaration.name,
                        "description": declaration.description or "",
                        "parameters": parameters
                        or {"type": "object", "properties": {}},
                    },
                }
            )
    return tools


def _rate_limit_retry_delay(error: RateLimitError | None) -> float:
    if error is None:
        return 1.0

    headers = error.response.headers
    for header in ("retry-after", "x-ratelimit-reset-tokens"):
        value = headers.get(header)
        if value:
            try:
                return max(float(value), 0.0)
            except ValueError:
                duration = _duration_seconds(value)
                if duration is not None:
                    return duration

    match = re.search(
        r"try again in ((?:\d+(?:\.\d+)?(?:ms|h|m|s))+)",
        str(error),
        re.IGNORECASE,
    )
    if match:
        duration = _duration_seconds(match.group(1).lower())
        if duration is not None:
            return duration
    return 1.0


def _duration_seconds(value: str) -> float | None:
    units = {"h": 3600, "m": 60, "s": 1, "ms": 0.001}
    parts = re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", value)
    if not parts or "".join(number + unit for number, unit in parts) != value:
        return None
    return sum(float(number) * units[unit] for number, unit in parts)
