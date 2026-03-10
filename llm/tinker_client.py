"""
Tinkr API client wrapping the OpenAI-compatible chat/completions endpoint.

Uses the `openai` SDK with a custom base_url pointing at Tinkr's inference service.
Requires TINKER_API_KEY in the environment.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError
except ImportError as e:
    raise ImportError(
        "openai package is required for live agent mode. Install with: pip install openai>=1.12.0"
    ) from e


TINKER_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
DEFAULT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"


@dataclass
class LLMResponse:
    """Structured response from a single LLM call."""

    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    logprobs: list[dict[str, float]] | None = None


class TinkerLLMClient:
    """
    Client for Tinkr's OpenAI-compatible inference API.

    Args:
        api_key:     Tinkr API key (falls back to TINKER_API_KEY env var).
        model:       Model identifier (HuggingFace name or tinker:// checkpoint path).
        temperature: Sampling temperature. Low = more deterministic.
        max_tokens:  Maximum tokens in the completion.
        timeout:     Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 800,
        timeout: float = 120.0,
    ):
        self.api_key = api_key or os.environ.get("TINKER_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set TINKER_API_KEY env var or pass api_key= explicitly."
            )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._client = OpenAI(
            base_url=TINKER_BASE_URL,
            api_key=self.api_key,
            timeout=timeout,
        )

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """
        Send a chat completion request and return structured response.

        Retries once on transient errors (timeout, rate limit, connection).
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_err: Exception | None = None
        for attempt in range(2):
            if attempt > 0:
                time.sleep(2.0)
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    logprobs=True,
                    top_logprobs=5,
                )
                choice = resp.choices[0]
                text = choice.message.content or ""

                meta: dict[str, Any] = {
                    "model": resp.model,
                    "finish_reason": choice.finish_reason,
                }
                if resp.usage:
                    meta["prompt_tokens"] = resp.usage.prompt_tokens
                    meta["completion_tokens"] = resp.usage.completion_tokens
                    meta["total_tokens"] = resp.usage.total_tokens
                if resp.id:
                    meta["request_id"] = resp.id

                logprobs_data = _extract_logprobs(choice)
                return LLMResponse(text=text, meta=meta, logprobs=logprobs_data)

            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                last_err = e
                continue
            except Exception:
                raise

        raise RuntimeError(f"Tinkr API failed after 2 attempts: {last_err}")

    def generate_patch(self, prompt: str) -> dict:
        """
        Convenience wrapper matching the spec: returns dict with text, patch, meta, logprobs.
        The caller is responsible for building the prompt via llm.prompting.
        """
        from llm.prompting import extract_diff

        resp = self.generate(prompt, system=_SYSTEM_MSG)
        patch = extract_diff(resp.text)
        return {
            "text": resp.text,
            "patch": patch,
            "meta": resp.meta,
            "logprobs": resp.logprobs,
        }


def _extract_logprobs(choice: Any) -> list[dict[str, float]] | None:
    """Pull top-k logprobs from OpenAI response if present; return None otherwise."""
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        return None
    content = getattr(lp, "content", None)
    if not content:
        return None
    result: list[dict[str, float]] = []
    for token_lp in content:
        pos: dict[str, float] = {}
        if hasattr(token_lp, "top_logprobs"):
            for entry in token_lp.top_logprobs:
                pos[entry.token] = entry.logprob
        elif hasattr(token_lp, "token") and hasattr(token_lp, "logprob"):
            pos[token_lp.token] = token_lp.logprob
        if pos:
            result.append(pos)
    return result if result else None


def validate_env() -> tuple[bool, str]:
    """Check that TINKER_API_KEY is set. Returns (ok, message)."""
    key = os.environ.get("TINKER_API_KEY", "")
    if not key:
        return False, "TINKER_API_KEY environment variable is not set."
    return True, f"TINKER_API_KEY is set ({len(key)} chars)."


_SYSTEM_MSG = (
    "You are a precise code-editing assistant. "
    "You will receive a buggy Python function, failing test output, and instructions. "
    "Output ONLY a unified diff (no explanations, no markdown fences, no commentary). "
    "If you cannot produce a diff, output exactly: NO_PATCH"
)


# ---------------------------------------------------------------------------
# Native Tinker SDK client (real logprobs via SamplingClient)
# ---------------------------------------------------------------------------

NATIVE_DEFAULT_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"


class TinkerNativeClient:
    """
    Native Tinker SDK client that returns real log-probabilities.

    Uses tinker.SamplingClient instead of the OpenAI-compatible endpoint.
    Requires the ``tinker`` and ``transformers`` packages.

    Two-call approach for top-k logprobs:
      1. ``sample()`` to generate tokens (returns top-1 logprob per token).
      2. ``sample()`` again with prompt+generated as input and
         ``include_prompt_logprobs=True`` to get top-k logprobs, then slice
         off the prompt portion.
    """

    def __init__(
        self,
        model: str = NATIVE_DEFAULT_MODEL,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ):
        try:
            import tinker
            from tinker import types as tinker_types
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "tinker and transformers packages are required for native client. "
                "Install with: pip install tinker transformers"
            ) from exc

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._tinker_types = tinker_types

        self._tokenizer = AutoTokenizer.from_pretrained(model)

        service_client = tinker.ServiceClient()
        training_client = service_client.create_lora_training_client(
            base_model=model, rank=8,
        )
        self._sampling_client = training_client.save_weights_and_get_sampling_client(
            name="drift_heatmap_sampler",
        )

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """
        Generate a completion with real top-k log-probabilities via two-call
        approach:
          1. Sample to get generated tokens.
          2. Prefill full sequence to get top-k logprobs for generated portion.
        """
        import sys
        tinker_types = self._tinker_types

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        result = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        if hasattr(result, "input_ids"):
            prompt_tokens = list(result.input_ids)
        elif isinstance(result, dict):
            prompt_tokens = list(result["input_ids"])
        else:
            prompt_tokens = list(result)

        # --- Step 1: sample to get generated tokens ---
        resp = self._sampling_client.sample(
            prompt=tinker_types.ModelInput.from_ints(prompt_tokens),
            num_samples=1,
            sampling_params=tinker_types.SamplingParams(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        ).result()

        seq = resp.sequences[0]
        gen_tokens = list(seq.tokens)
        text = self._tokenizer.decode(gen_tokens, skip_special_tokens=True)

        meta: dict[str, Any] = {
            "model": self.model,
            "client": "tinker_native",
            "stop_reason": seq.stop_reason,
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": len(gen_tokens),
        }

        # --- Step 2: prefill full sequence for top-k logprobs ---
        logprobs_data: list[dict[str, float]] | None = None
        try:
            full_sequence = prompt_tokens + gen_tokens
            logprob_resp = self._sampling_client.sample(
                prompt=tinker_types.ModelInput.from_ints(full_sequence),
                num_samples=1,
                sampling_params=tinker_types.SamplingParams(max_tokens=1),
                include_prompt_logprobs=True,
                topk_prompt_logprobs=5,
            ).result()

            all_topk = logprob_resp.topk_prompt_logprobs
            gen_topk = all_topk[len(prompt_tokens):]

            logprobs_data = []
            for position_topk in gen_topk:
                if position_topk is None:
                    continue
                pos_dict: dict[str, float] = {}
                for token_id, logprob in position_topk:
                    token_str = self._tokenizer.decode([token_id])
                    pos_dict[token_str] = logprob
                if pos_dict:
                    logprobs_data.append(pos_dict)

            if not logprobs_data:
                logprobs_data = None

        except Exception as e:
            print(f"  [logprobs prefill failed: {e}]", file=sys.stderr)
            logprobs_data = None

        return LLMResponse(text=text, meta=meta, logprobs=logprobs_data)

    def generate_patch(self, prompt: str) -> dict:
        """Convenience wrapper matching TinkerLLMClient.generate_patch interface."""
        from llm.prompting import extract_diff

        resp = self.generate(prompt, system=_SYSTEM_MSG)
        patch = extract_diff(resp.text)
        return {
            "text": resp.text,
            "patch": patch,
            "meta": resp.meta,
            "logprobs": resp.logprobs,
        }
