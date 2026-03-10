"""LLM integration for live agent mode (Tinkr API via OpenAI-compatible endpoint)."""

from llm.tinker_client import TinkerLLMClient, TinkerNativeClient
from llm.prompting import build_patch_prompt, extract_diff, extract_complete_function
