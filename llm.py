"""
Thin wrapper around the Anthropic API for the three places this project
uses a language model.

Where the LLM is allowed to act, and why, follows from what the 2023-2026
literature actually supports (docs/research-llm-agents.md):

- It may PROPOSE factor expressions, which a deterministic backtester then
  judges (AlphaAgent, KDD 2025; RD-Agent(Q), NeurIPS 2025). The model never
  sees prices, dates or tickers, so it cannot recall what happened.
- It may NARRATE numbers that have already been computed. Geng et al. (2026)
  found LLMs reproduce a supplied ranking reliably but misalign badly when
  asked to infer one, so it is handed the ranking and forbidden new numbers.
- It may CRITICISE a rebalance whose checks were computed in Python, without
  the power to relax any of them (TradeTrap, 2025; OpenPM, 2026).

It may never score a symbol, size a position, or place a trade. Every
replication of the LLM-trader papers that controlled for the model's
training cutoff - FinSABER, Profit Mirage, StockBench, The Alpha Illusion -
found the edge at or below buy-and-hold.

The API key comes from ANTHROPIC_API_KEY or an `ant auth login` profile.
Absent either, callers get LLMUnavailable and every feature here degrades to
its deterministic half.
"""

import json
import logging
import os

logger = logging.getLogger('stock_app.llm')

MODEL = os.environ.get('STOCK_LLM_MODEL', 'claude-opus-5')
MAX_TOKENS = 16000

_client = None


class LLMUnavailable(RuntimeError):
    """No SDK, no credentials, or the call failed."""


def available():
    """True if a call has a chance of succeeding.

    Constructing a client succeeds with no credentials at all - the SDK only
    resolves them when a request is made - so this checks that something
    resolvable is actually present rather than that the object exists.
    """
    try:
        client = _get_client()
    except LLMUnavailable:
        return False
    if any(os.environ.get(k) for k in
           ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_PROFILE',
            'ANTHROPIC_IDENTITY_TOKEN', 'ANTHROPIC_IDENTITY_TOKEN_FILE')):
        return True
    # an `ant auth login` profile on disk also counts
    if getattr(client, 'api_key', None) or getattr(client, 'auth_token', None):
        return True
    profile = os.path.join(os.path.expanduser('~'), '.config', 'anthropic')
    return os.path.isdir(profile)


def _get_client():
    global _client
    if _client is not None:
        return _client
    try:
        import anthropic
    except ImportError as e:
        raise LLMUnavailable("the anthropic package is not installed") from e
    try:
        _client = anthropic.Anthropic()
    except Exception as e:
        raise LLMUnavailable(f"could not construct a client: {e}") from e
    return _client


def structured(system, prompt, schema, effort='high', model=None,
               max_tokens=MAX_TOKENS):
    """One call that must come back as JSON matching `schema`.

    Structured output is not decoration here. Every caller in this project
    post-validates the result against numbers it computed itself, and a
    free-text response would have to be parsed before it could be checked.
    """
    client = _get_client()
    try:
        response = client.messages.create(
            model=model or MODEL,
            max_tokens=max_tokens,
            system=system,
            thinking={'type': 'adaptive'},
            output_config={
                'effort': effort,
                'format': {'type': 'json_schema', 'schema': schema},
            },
            messages=[{'role': 'user', 'content': prompt}],
        )
    except Exception as e:
        raise LLMUnavailable(f"request failed: {e}") from e

    if getattr(response, 'stop_reason', None) == 'refusal':
        detail = getattr(response, 'stop_details', None)
        raise LLMUnavailable(
            f"request refused ({getattr(detail, 'category', 'unspecified')})")

    text = next((b.text for b in response.content if b.type == 'text'), None)
    if not text:
        raise LLMUnavailable("no text block in the response")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMUnavailable(f"response was not valid JSON: {e}") from e

    usage = getattr(response, 'usage', None)
    if usage is not None:
        logger.info("llm call: %s in, %s out",
                    getattr(usage, 'input_tokens', '?'),
                    getattr(usage, 'output_tokens', '?'))
    return payload
