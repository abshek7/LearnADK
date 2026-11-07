"""Tests for structured OpenRouter responses used by DeepEval."""

import asyncio
from importlib.util import find_spec
import os
import unittest
from unittest.mock import patch

import httpx
from pydantic import BaseModel

if find_spec("deepeval") is not None:
    import evaluations.evaluate_agent as evaluation
else:
    evaluation = None


@unittest.skipIf(evaluation is None, "Install requirements-eval.txt to run these tests.")
class OpenRouterEvaluationModelTests(unittest.TestCase):
    class TruthsSchema(BaseModel):
        truths: list[str]

    def setUp(self) -> None:
        assert evaluation is not None
        self.environment = patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "DEEPEVAL_MODEL": "nvidia/nemotron-3.5-super-vl:free",
                "DEEPEVAL_FALLBACK_MODELS": (
                    "xingchen-agi/xing4.0-29b-a4b:free"
                ),
                "DEEPEVAL_MAX_COMPLETION_TOKENS": "128",
            },
        )
        self.environment.start()
        self.model = evaluation.OpenRouterEvaluationModel()
        self.request = httpx.Request(
            "POST",
            evaluation.OPENROUTER_CHAT_COMPLETIONS_URL,
        )

    def tearDown(self) -> None:
        self.model.http_client.close()
        self.environment.stop()

    def _response(
        self,
        content: str,
        *,
        finish_reason: str = "stop",
    ) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": content},
                        "finish_reason": finish_reason,
                    }
                ]
            },
            request=self.request,
        )

    def test_invalid_json_retries_with_free_fallback(self) -> None:
        with patch.object(
            self.model.http_client,
            "post",
            side_effect=[
                self._response("This is not JSON"),
                self._response('```json\n{"truths":["supported"]}\n```'),
            ],
        ) as post:
            result = self.model.generate(
                "Extract facts",
                schema=self.TruthsSchema,
            )

        self.assertEqual(
            self.TruthsSchema.model_validate_json(result).truths,
            ["supported"],
        )
        self.assertEqual(
            [call.kwargs["json"]["model"] for call in post.call_args_list],
            [self.model.model_id, *self.model.fallback_models],
        )
        self.assertEqual(
            self.model.successful_models,
            [self.model.fallback_models[0]],
        )
        self.assertEqual(len(self.model.structured_output_errors), 1)

    def test_truncated_output_retries_with_free_fallback(self) -> None:
        with patch.object(
            self.model.http_client,
            "post",
            side_effect=[
                self._response('{"truths":[', finish_reason="length"),
                self._response('{"truths":["supported"]}'),
            ],
        ):
            result = self.model.generate(
                "Extract facts",
                schema=self.TruthsSchema,
            )

        self.assertEqual(
            self.TruthsSchema.model_validate_json(result).truths,
            ["supported"],
        )
        self.assertIn("truncated", self.model.structured_output_errors[0]["error"])

    def test_invalid_json_from_all_models_has_actionable_error(self) -> None:
        with patch.object(
            self.model.http_client,
            "post",
            side_effect=lambda *args, **kwargs: self._response("not JSON"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "failed to return valid structured output",
            ):
                self.model.generate(
                    "Extract facts",
                    schema=self.TruthsSchema,
                )

        self.assertEqual(len(self.model.structured_output_errors), 6)
        self.assertEqual(self.model.successful_models, [])

    def test_async_generation_validates_structured_output(self) -> None:
        response = self._response('{"truths":["supported"]}')

        class FakeAsyncClient:
            def __init__(self, **_: object) -> None:
                pass

            async def __aenter__(self) -> "FakeAsyncClient":
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

            async def post(self, url: str, json: dict[str, object]) -> httpx.Response:
                return response

        assert evaluation is not None
        with patch.object(evaluation.httpx, "AsyncClient", FakeAsyncClient):
            result = asyncio.run(
                self.model.a_generate(
                    "Extract facts",
                    schema=self.TruthsSchema,
                )
            )

        self.assertEqual(
            self.TruthsSchema.model_validate_json(result).truths,
            ["supported"],
        )
