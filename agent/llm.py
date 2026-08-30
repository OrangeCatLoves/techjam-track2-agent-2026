"""The LLM connection, and the token meter that decides the Feasibility score.

OWNS
    - every call to a language model, and the accounting for it
    - the per-run token ceiling, enforced rather than advised
    - deterministic mode (``LLM_PROVIDER=none``), which must work from day one
    - screening every prompt before it leaves the process

MUST NEVER
    - count only successful calls. Failed requests, retries and timeouts consume
      tokens. A counter that increments on success alone reports a Feasibility
      number wrong by an unknown amount, and Feasibility is 15% of the grade,
      scored only if we beat the baseline
    - send a hidden-test metric to a third party. Prompts are built from screened
      diagnostics, and are screened again here, because this is the last point at
      which the text is still ours
    - log an API key, or any part of one
    - retry forever. A loop stuck re-proposing at 4am must stop, not drain the
      account

DESIGN NOTE ON FAILED CALLS
    When a request raises, the response object usually never arrives, so the true
    token count is unknown. Charging zero would understate the bill; the meter
    therefore charges an *estimate* from the prompt length, marks the call as
    estimated, and reports how many calls were estimated. An honest approximation
    that can only overstate is worth more than a precise number that is wrong.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List

from harness import data as hdata
from harness import guards

#: Roughly four characters per token for English prose and code. Used only to
#: estimate the cost of a call that failed before reporting its own usage.
CHARS_PER_TOKEN = 4

FAST, STRONG = 'fast', 'strong'


class LLMError(RuntimeError):
    """A call failed after every retry."""


class TokenBudgetExceeded(RuntimeError):
    """The per-run ceiling was reached. Not a crash: a stop."""


class LLMUnavailable(RuntimeError):
    """No provider is configured. Deterministic mode should be used instead."""


@dataclass
class LLMResponse:
    """One call, whether or not it succeeded."""
    text: str
    model: str
    role: str                      # 'fast' or 'strong'
    input_tokens: int
    output_tokens: int
    estimated: bool                # True if the counts are inferred, not reported
    seconds: float
    attempts: int
    stop_reason: str | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenBudget:
    """The meter. Charged on every call, successful or not."""
    limit: int = 300_000
    warn_at: int = 200_000
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    failed_calls: int = 0
    estimated_calls: int = 0
    by_model: Dict[str, int] = field(default_factory=dict)
    warned: bool = False

    @property
    def spent(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.spent)

    @property
    def exceeded(self) -> bool:
        return self.spent >= self.limit

    def charge(self, response: LLMResponse) -> None:
        """Record a call. Called for failures too; that is the point."""
        self.input_tokens += max(0, response.input_tokens)
        self.output_tokens += max(0, response.output_tokens)
        self.calls += 1
        if not response.ok:
            self.failed_calls += 1
        if response.estimated:
            self.estimated_calls += 1
        self.by_model[response.model] = (
            self.by_model.get(response.model, 0) + response.total_tokens)

    def check(self) -> None:
        """Raise if the ceiling is reached. Callers treat this as a stop signal."""
        if self.exceeded:
            raise TokenBudgetExceeded(
                f'token ceiling reached: {self.spent:,} of {self.limit:,} spent '
                f'across {self.calls} call(s). Raise agent.token_budget.limit in '
                f'configs/base.yaml if this run is meant to cost more.')

    def should_warn(self) -> bool:
        """True once, when the warning threshold is first crossed."""
        if not self.warned and self.spent >= self.warn_at:
            self.warned = True
            return True
        return False

    def as_dict(self) -> Dict[str, Any]:
        return {'input': self.input_tokens, 'output': self.output_tokens,
                'total': self.spent, 'limit': self.limit,
                'remaining': self.remaining, 'calls': self.calls,
                'failed_calls': self.failed_calls,
                'estimated_calls': self.estimated_calls,
                'by_model': dict(self.by_model)}


def estimate_tokens(text: str) -> int:
    """Rough token count, for charging a call that never reported its own."""
    return max(1, len(text or '') // CHARS_PER_TOKEN)


# --------------------------------------------------------------------------
# the client
# --------------------------------------------------------------------------

class LLMClient:
    """Every language-model call in the project goes through here.

    *transport* is injectable so the whole loop can be tested without spending
    money or needing a network. The default transport is built lazily, so
    importing this module never requires an API key.
    """

    def __init__(self, *,
                 provider: str | None = None,
                 models: Dict[str, str] | None = None,
                 budget: TokenBudget | None = None,
                 transport: Callable[..., Any] | None = None,
                 max_retries: int = 3,
                 backoff_seconds: float = 2.0,
                 on_event: Callable[[str, str, Dict[str, Any]], None] | None = None):
        env = _environment()
        self.provider = (provider or env.get('LLM_PROVIDER') or 'none').strip().lower()
        self.models = models or {
            FAST: env.get('LLM_MODEL_FAST') or '',
            STRONG: env.get('LLM_MODEL_STRONG') or '',
        }
        self.budget = budget or budget_from_config()
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.calls: List[LLMResponse] = []
        self._transport = transport
        self._on_event = on_event

    # -- availability ------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """False in deterministic mode. Callers must have a path for that."""
        return self.provider not in ('none', '', 'off', 'deterministic')

    def model_for(self, role: str) -> str:
        if role not in (FAST, STRONG):
            raise ValueError(f"role must be {FAST!r} or {STRONG!r}, got {role!r}")
        name = self.models.get(role) or ''
        if not name:
            raise LLMUnavailable(
                f'no model configured for role {role!r}. Set LLM_MODEL_'
                f'{role.upper()} in .env, or run with LLM_PROVIDER=none.')
        return name

    # -- the call ----------------------------------------------------------

    def complete(self, prompt: str, *, system: str | None = None,
                 role: str = FAST, max_tokens: int = 2048,
                 temperature: float = 0.0) -> LLMResponse:
        """One completion. Charged to the budget whatever happens.

        Raises ``TokenBudgetExceeded`` *before* spending if the ceiling is already
        reached, and ``LLMError`` if every attempt failed.
        """
        if not self.enabled:
            raise LLMUnavailable(
                'LLM_PROVIDER is none; use the deterministic path instead of '
                'calling complete().')

        # The last point at which this text is still ours.
        guards.assert_no_test_metrics(prompt, where='LLM prompt')
        if system:
            guards.assert_no_test_metrics(system, where='LLM system prompt')

        self.budget.check()
        model = self.model_for(role)
        estimated_input = estimate_tokens(prompt) + estimate_tokens(system or '')

        started = time.time()
        last_error: str | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self._send(model=model, prompt=prompt, system=system,
                                 max_tokens=max_tokens, temperature=temperature)
                response = _to_response(raw, model=model, role=role,
                                        seconds=time.time() - started,
                                        attempts=attempt,
                                        fallback_input=estimated_input)
                self._record(response)
                return response
            except Exception as exc:                  # noqa: BLE001 - reported, not raised
                last_error = f'{type(exc).__name__}: {exc}'
                # Charge the attempt. A failed request still consumed input
                # tokens, and pretending otherwise understates the bill.
                self._record(LLMResponse(
                    text='', model=model, role=role,
                    input_tokens=estimated_input, output_tokens=0, estimated=True,
                    seconds=time.time() - started, attempts=attempt,
                    error=last_error))
                self._emit('llm_retry' if attempt < self.max_retries else 'llm_failed',
                           last_error, {'model': model, 'attempt': attempt})
                try:
                    self.budget.check()
                except TokenBudgetExceeded:
                    raise
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * attempt)

        raise LLMError(f'all {self.max_retries} attempts failed. Last: {last_error}')

    def _record(self, response: LLMResponse) -> None:
        self.calls.append(response)
        self.budget.charge(response)
        if self.budget.should_warn():
            self._emit('token_warning',
                       f'{self.budget.spent:,} of {self.budget.limit:,} tokens spent',
                       self.budget.as_dict())

    def _emit(self, kind: str, message: str, fields: Dict[str, Any]) -> None:
        if self._on_event is not None:
            self._on_event(kind, message, fields)

    # -- transport ---------------------------------------------------------

    def _send(self, *, model: str, prompt: str, system: str | None,
              max_tokens: int, temperature: float):
        if self._transport is None:
            self._transport = build_transport(self.provider)
        return self._transport(model=model, prompt=prompt, system=system,
                               max_tokens=max_tokens, temperature=temperature)

    # -- reporting ---------------------------------------------------------

    def usage(self) -> Dict[str, Any]:
        """The Feasibility numbers, with the estimated share stated."""
        report = self.budget.as_dict()
        report['provider'] = self.provider
        report['models'] = dict(self.models)
        report['note'] = (
            f"{report['estimated_calls']} of {report['calls']} call(s) had their "
            f'token count estimated from prompt length because the request failed '
            f'before reporting usage. Estimates can only overstate.')
        guards.assert_record_clean(report, where='llm usage')
        return report


# --------------------------------------------------------------------------
# transports
# --------------------------------------------------------------------------

def _environment() -> Dict[str, str]:
    """Environment first, then ``.env``. Values are never logged."""
    merged = dict(hdata._dotenv())
    for key in ('LLM_PROVIDER', 'LLM_MODEL_FAST', 'LLM_MODEL_STRONG',
                'ANTHROPIC_API_KEY'):
        value = os.environ.get(key)
        if value:
            merged[key] = value
    return merged


#: Provider names that route through the Claude Code CLI rather than the API.
CLI_PROVIDERS = ('claude_cli', 'claude-cli', 'claude_code', 'claude-code', 'cli')

#: Environment variables that make the Claude Code CLI abandon the subscription.
#: Verified: with either set, it prints "ANTHROPIC_API_KEY ... takes precedence
#: over your claude.ai login" and bills the API account instead. A dead key in
#: .env would therefore turn every call into a credit-balance error.
SUBSCRIPTION_BLOCKERS = ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN',
                         'ANTHROPIC_BASE_URL', 'ANTHROPIC_BEDROCK_BASE_URL',
                         'CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX')

#: Tools the CLI must not use. We want a completion, not an agent with filesystem
#: access: this process already owns the sandbox, the guards and the patch
#: validator, and a second agent editing files behind them would bypass all three.
CLI_DISALLOWED_TOOLS = ('Bash,Read,Write,Edit,NotebookEdit,Glob,Grep,'
                        'WebSearch,WebFetch,Task,TodoWrite')


class _CLIResponse:
    """Adapts the CLI's JSON to the shape ``_to_response`` already understands."""

    def __init__(self, payload: Dict[str, Any], model: str):
        self.content = payload.get('result') or ''
        self.model = model
        self.stop_reason = payload.get('stop_reason')
        usage = payload.get('usage') or {}
        # Cache creation and cache reads are real tokens and are counted. The
        # Feasibility figure should not flatter us by ignoring the ones Claude
        # Code spends on its own context.
        self.usage = type('Usage', (), {
            'input_tokens': int(usage.get('input_tokens', 0) or 0)
                            + int(usage.get('cache_creation_input_tokens', 0) or 0)
                            + int(usage.get('cache_read_input_tokens', 0) or 0),
            'output_tokens': int(usage.get('output_tokens', 0) or 0),
        })()
        self.cost_usd = payload.get('total_cost_usd')
        self.is_error = bool(payload.get('is_error'))


def _claude_cli_transport(timeout: float = 300.0) -> Callable[..., Any]:
    """Route calls through the Claude Code CLI, on the user's subscription.

    No API key, and none permitted: ``SUBSCRIPTION_BLOCKERS`` are stripped from
    the child environment, because their presence silently switches billing away
    from the subscription and onto an API account that may have no balance.

    The prompt goes over **stdin**, not as an argument. A proposer prompt is
    ~15,000 characters and Windows caps a command line at about 32,000, so an
    argument would work right up until the corpus grew.
    """
    import json as _json
    import shutil
    import subprocess

    executable = shutil.which('claude')
    if not executable:
        raise LLMUnavailable(
            'the `claude` CLI is not on PATH. Install Claude Code, or set '
            'LLM_PROVIDER=anthropic with an API key that has credit.')

    def send(*, model: str, prompt: str, system: str | None,
             max_tokens: int, temperature: float):
        argv = [executable, '-p', '--model', model, '--output-format', 'json',
                '--disallowed-tools', CLI_DISALLOWED_TOOLS]
        if system:
            argv += ['--system-prompt', system]

        env = {k: v for k, v in os.environ.items()
               if k not in SUBSCRIPTION_BLOCKERS}
        env['PYTHONIOENCODING'] = 'utf-8'

        proc = subprocess.run(argv, input=prompt, env=env, timeout=timeout,
                              capture_output=True, text=True,
                              encoding='utf-8', errors='replace')
        if proc.returncode != 0:
            raise LLMError(f'claude CLI exited {proc.returncode}: '
                           f'{(proc.stderr or proc.stdout or "")[-500:]}')

        # A warning line can precede the JSON, so parse from the first brace.
        text = proc.stdout or ''
        start = text.find('{')
        if start == -1:
            raise LLMError(f'no JSON in CLI output: {text[:300]}')
        try:
            payload = _json.loads(text[start:])
        except _json.JSONDecodeError as exc:
            raise LLMError(f'could not parse CLI output: {exc}') from None

        response = _CLIResponse(payload, model)
        if response.is_error or not response.content:
            raise LLMError(
                f'the CLI returned an error or empty result '
                f'(stop_reason={response.stop_reason}): {str(response.content)[:300]}')
        return response

    return send


def build_transport(provider: str) -> Callable[..., Any]:
    """Pick a transport for *provider*."""
    if provider in CLI_PROVIDERS:
        return _claude_cli_transport()
    return _anthropic_transport()


def _anthropic_transport() -> Callable[..., Any]:
    """Build the real client lazily, so importing this module needs no key."""
    import anthropic

    key = _environment().get('ANTHROPIC_API_KEY')
    if not key:
        raise LLMUnavailable(
            'ANTHROPIC_API_KEY is not set. Put it in .env (which is gitignored) '
            'and never in .env.example, which is tracked.')
    client = anthropic.Anthropic(api_key=key)

    # The SDK's accepted parameters shift between versions: anthropic 1.2.0
    # dropped `temperature` from messages.create entirely. Rather than pin a
    # version or guess, ask the installed SDK what it takes and send only that.
    import inspect
    accepted = set(inspect.signature(client.messages.create).parameters)

    def send(*, model: str, prompt: str, system: str | None,
             max_tokens: int, temperature: float):
        kwargs: Dict[str, Any] = {
            'model': model, 'max_tokens': max_tokens,
            'messages': [{'role': 'user', 'content': prompt}],
        }
        if system and 'system' in accepted:
            kwargs['system'] = system
        if 'temperature' in accepted:
            kwargs['temperature'] = temperature
        return client.messages.create(**kwargs)

    return send


def _to_response(raw: Any, *, model: str, role: str, seconds: float,
                 attempts: int, fallback_input: int) -> LLMResponse:
    """Normalise a provider response, tolerating a shape that shifts under us."""
    text = ''
    content = getattr(raw, 'content', None)
    if isinstance(content, str):
        text = content
    elif content:
        parts = []
        for block in content:
            piece = getattr(block, 'text', None)
            if piece is None and isinstance(block, dict):
                piece = block.get('text')
            if piece:
                parts.append(piece)
        text = ''.join(parts)
    elif isinstance(raw, str):
        text = raw

    usage = getattr(raw, 'usage', None)
    input_tokens = getattr(usage, 'input_tokens', None) if usage else None
    output_tokens = getattr(usage, 'output_tokens', None) if usage else None
    estimated = input_tokens is None or output_tokens is None

    return LLMResponse(
        text=text,
        model=getattr(raw, 'model', model) or model,
        role=role,
        input_tokens=int(input_tokens) if input_tokens is not None else fallback_input,
        output_tokens=(int(output_tokens) if output_tokens is not None
                       else estimate_tokens(text)),
        estimated=estimated,
        seconds=seconds,
        attempts=attempts,
        stop_reason=getattr(raw, 'stop_reason', None),
    )


def budget_from_config() -> TokenBudget:
    """Ceiling and warning threshold from ``configs/base.yaml``."""
    cfg = {}
    try:
        cfg = (hdata.load_config().get('agent', {}) or {}).get('token_budget', {}) or {}
    except Exception:
        cfg = {}
    return TokenBudget(limit=int(cfg.get('limit', 300_000)),
                       warn_at=int(cfg.get('warn_at', 200_000)))
