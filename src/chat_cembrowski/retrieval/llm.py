"""
LLM provider configuration for the retrieval path.

The query engine talks to its model through the OpenAI SDK, and both providers
we care about speak that same chat-completions schema — OpenRouter natively, and
OpenAI directly. So swapping providers is a `base_url` and a model string, not
an abstraction layer: the SDK already is the abstraction layer. This module
holds the two-line difference between them plus the per-provider defaults.

Only the retrieval path is routed this way. `data/` still calls OpenAI directly
for OCR (`data/ocr.py`) and metadata extraction (`data/ingestion.py`), so
`OPENAI_API_KEY` and the `openai` dependency are required regardless of what
`LLM_PROVIDER` is set to.

Environment:
    LLM_PROVIDER           "openrouter" (default) or "openai"
    OPENROUTER_API_KEY     required when provider is "openrouter"
    OPENAI_API_KEY         required when provider is "openai"
    CHAT_MODEL             overrides the synthesis model for the provider
    CLASSIFIER_MODEL       overrides the classify/condense model
    LLM_REASONING_EFFORT   thinking budget for synthesis (see below)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)
load_dotenv()


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_PROVIDER = "openrouter"

# (chat_model, classifier_model) per provider.
#
# On OpenRouter these are Gemini's mid and low tiers, not the flagship. The
# synthesis job here is grounded summarization over ~5k tokens of already
# retrieved context with strict citation discipline — it is not reasoning-bound,
# so Pro's premium buys little (see README, "Why these models"). The classifier
# emits a single word against a prompt that already spells out the decision rule
# with paired examples, which Flash-Lite handles for roughly a tenth the price.
DEFAULT_MODELS: dict[str, tuple[str, str]] = {
    "openrouter": ("google/gemini-3.6-flash", "google/gemini-3.5-flash-lite"),
    "openai": ("gpt-4.1", "gpt-4.1-mini"),
}

# Values OpenRouter accepts for `reasoning.effort`. Gemini 3 models use Google's
# own `thinkingLevel` internally; OpenRouter maps "minimal"/"low"/"medium"
# straight across and collapses "high"/"xhigh" onto Gemini's "high".
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")

DEFAULT_REASONING_EFFORT = "minimal"

# The classifier's effort is pinned rather than configurable. It is a blocking
# call on the critical path of every single question, and the task — pick one of
# three labels — has no intermediate steps for reasoning to help with. Letting a
# global LLM_REASONING_EFFORT=high leak into it would add seconds to first token
# on every query for no accuracy gain. Same argument applies to condensing a
# follow-up into a standalone question.
CLASSIFIER_REASONING_EFFORT = "minimal"

# Sent by OpenRouter convention for attribution; harmless if the dashboard
# ranking is not something we care about.
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/rainleaforg/chat-cembrowski",
    "X-Title": "chat-cembrowski",
}


@dataclass(frozen=True)
class LLMConfig:
    """Which provider and models the retrieval path should use."""

    provider: str
    chat_model: str
    classifier_model: str
    reasoning_effort: str

    @property
    def supports_reasoning_control(self) -> bool:
        """
        Whether this provider accepts OpenRouter's `reasoning` request field.

        OpenAI rejects unknown body fields, so the parameter has to be omitted
        entirely rather than sent with a neutral value.
        """
        return self.provider == "openrouter"


def get_config() -> LLMConfig:
    """
    Build the LLM configuration from the environment.

    Raises ValueError on an unrecognized LLM_PROVIDER rather than silently
    falling back — a typo'd provider should fail loudly at startup, not send
    production traffic somewhere unintended.
    """
    provider = os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(
            f"LLM_PROVIDER must be one of {sorted(DEFAULT_MODELS)}, got '{provider}'."
        )

    default_chat, default_classifier = DEFAULT_MODELS[provider]

    effort = os.getenv("LLM_REASONING_EFFORT", DEFAULT_REASONING_EFFORT).strip().lower()
    if effort not in REASONING_EFFORTS:
        logger.warning(
            f"LLM_REASONING_EFFORT='{effort}' is not one of {REASONING_EFFORTS}; "
            f"falling back to '{DEFAULT_REASONING_EFFORT}'."
        )
        effort = DEFAULT_REASONING_EFFORT

    return LLMConfig(
        provider=provider,
        chat_model=os.getenv("CHAT_MODEL") or default_chat,
        classifier_model=os.getenv("CLASSIFIER_MODEL") or default_classifier,
        reasoning_effort=effort,
    )


def get_llm_client(config: LLMConfig | None = None) -> OpenAI:
    """
    Return an OpenAI-SDK client pointed at the configured provider.

    OpenRouter exposes an OpenAI-compatible endpoint, so the only difference
    from a direct OpenAI client is the base URL, the key, and the attribution
    headers.
    """
    config = config or get_config()

    if config.provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not set in environment "
                "(required when LLM_PROVIDER='openrouter')."
            )
        logger.info(
            f"Using OpenRouter (chat='{config.chat_model}', "
            f"classifier='{config.classifier_model}')."
        )
        return OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers=OPENROUTER_HEADERS,
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set in environment "
            "(required when LLM_PROVIDER='openai')."
        )
    logger.info(
        f"Using OpenAI directly (chat='{config.chat_model}', "
        f"classifier='{config.classifier_model}')."
    )
    return OpenAI(api_key=api_key)


def completion_extras(config: LLMConfig, effort: str | None = None) -> dict:
    """
    Extra `chat.completions.create` kwargs controlling the thinking budget.

    Returns `{"extra_body": {"reasoning": {...}}}` on OpenRouter and `{}` on
    OpenAI, so call sites can splat it unconditionally:

        self.llm.chat.completions.create(..., **llm.completion_extras(cfg))

    This matters more than it looks. Gemini 2.5+ and the whole 3.x line bill
    reasoning tokens as output AND count them against `max_tokens`, so an
    unbudgeted thinking model can burn its entire allowance before emitting a
    visible character — returning empty content with no exception raised. Every
    call site pairs this with a `max_tokens` generous enough that reasoning
    cannot starve the response.
    """
    if not config.supports_reasoning_control:
        return {}
    return {"extra_body": {"reasoning": {"effort": effort or config.reasoning_effort}}}
