"""Choosing the next experiment, and writing the code for it.

OWNS
    - the prompt that asks for one experiment, given measured facts
    - validation of what comes back, against the frozen spec in CLAUDE.md 11.1
    - the deterministic fallback, which must produce a valid run with no LLM at all

MUST NEVER
    - execute a queue. ``knowledge/methods.md`` is reference material the proposer
      retrieves from; it is not a list of configurations to work through. An agent
      running a human's checklist scores badly on both Innovation and Impact,
      however good the resulting model is (CLAUDE.md section 6.2)
    - accept a proposal whose content hash is already in the tried-set
    - let a hidden-test metric into a prompt. The corpus and the diagnosis are both
      screened before they are assembled, and the assembled prompt is screened
      again by ``llm.complete``

ON THE DETERMINISTIC PATH
    ``LLM_PROVIDER=none`` must produce a valid submission with no model call. It is
    insurance against an API outage during the scored run, and it is also the
    honest control: if the deterministic sequence scores as well as the agent, the
    agent was not adding anything, and we would rather know.

    The deterministic path is explicitly *not* the agent. It is labelled as such in
    every log record it produces, so a reader can never mistake one for the other.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from agent import llm as agent_llm
from harness import analyse as hanalyse
from harness import data as hdata
from harness import guards
from harness import losses as hlosses
from harness import patch as hpatch

#: The stages an experiment may target. From CLAUDE.md 11.1, interface 4.
TARGET_STAGES = ('objective', 'model', 'features', 'sampling', 'ensemble')

MAX_PATCH_CHARS = 12_000


class ProposalError(ValueError):
    """The proposal was malformed, forbidden, or already tried."""


@dataclass
class Proposal:
    """One experiment, as the agent specified it."""
    hypothesis: str
    target_stage: str
    patch_kind: str
    patch: str
    expected_gain: float = 0.0
    expected_cost_minutes: float = 0.0
    source: str = 'llm'                 # 'llm' or 'deterministic'
    analyses_requested: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def content_hash(self) -> str:
        return hpatch.content_hash(self.patch)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def as_record(self) -> Dict[str, Any]:
        """The log view: everything except the patch body, which is archived."""
        record = self.as_dict()
        record.pop('patch')
        record['content_hash'] = self.content_hash
        record['patch_chars'] = len(self.patch)
        return record


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def validate(proposal: Proposal, *, tried: List[str] | None = None) -> Proposal:
    """Check a proposal before any of it is written to disk."""
    if not proposal.hypothesis or len(proposal.hypothesis) < 20:
        raise ProposalError(
            'a proposal needs a real hypothesis: what is believed to be wrong, and '
            'why this change should move the metric')
    if proposal.target_stage not in TARGET_STAGES:
        raise ProposalError(
            f'target_stage {proposal.target_stage!r} is not one of {TARGET_STAGES}')
    if not proposal.patch.strip():
        raise ProposalError('a proposal must carry the code that implements it')
    if len(proposal.patch) > MAX_PATCH_CHARS:
        raise ProposalError(
            f'patch is {len(proposal.patch)} characters, over the '
            f'{MAX_PATCH_CHARS} limit')
    if 'CONFIG' not in proposal.patch:
        raise ProposalError(
            'a patch must define CONFIG, a dict of train_fm keyword arguments. '
            'An empty CONFIG is legal and trains the reference configuration.')
    if tried and proposal.content_hash in tried:
        raise ProposalError(
            f'this experiment has already been tried (content hash '
            f'{proposal.content_hash}). Propose something different.')

    report = hpatch.validate_source(
        proposal.patch, hdata.repo_root() / 'harness' / 'models' / 'gen' / 'probe.py')
    if not report.ok:
        raise ProposalError('the patch does not pass validation:\n  - '
                            + '\n  - '.join(report.reasons))
    return proposal


# --------------------------------------------------------------------------
# the prompt
# --------------------------------------------------------------------------

SYSTEM = """You are an autonomous ML research agent working on a within-user ranking
task. You propose ONE experiment at a time, write the Python for it yourself, read
the measured result, and decide what to try next.

You are NOT in an interactive session. There are no tools, no files to edit, and no
human reading your prose. Your entire output is parsed by a program. Do not describe
what you would do, do not apologise for missing tools, and do not write a report.

OUTPUT: exactly one JSON object. No prose before it, no prose after it, no markdown
fence. The `patch` field carries your Python source as a JSON string.

Rules you cannot break:
- Every number you are given was measured by the harness. Never invent one, and
  never restate one you were not given.
- Your patch is ONE Python module written to harness/models/gen/. It must define
  CONFIG. It must not define main(), argparse, __main__, or its own training loop:
  the harness runs the training. You supply the objective and the configuration.
- You may import numpy, pandas, scipy, sklearn, lightgbm, math, itertools,
  collections, and harness.losses. Nothing else. No os, no sys, no file access.
- A pairwise or listwise objective MUST build its pairs or lists WITHIN a group.
  The harness verifies this by permuting the grouping and requiring the loss to
  change; a loss that ignores `groups` is rejected.
- Never repeat an experiment already in the tried list.

Keep the hypothesis under 120 words. Reasoning that does not fit in the hypothesis
field is lost, so put it there rather than outside the JSON."""


#: A complete, valid patch, shown because the first real call invented its own
#: argparse runner: the contract was stated but never demonstrated, so the model
#: guessed an interface and the proposal was unusable.
#:
#: DELIBERATELY A WEAK IDEA. It is a per-list-normalised *pointwise* loss, which
#: is not a ranking objective at all. It demonstrates the mechanics -- how to
#: register, how to walk groups, what to return, what CONFIG looks like -- and
#: nothing about what to try. An example that showed a pairwise or listwise
#: objective would hand the agent the two ideas the whole exercise is asking it to
#: find, and Innovation is 20% of the grade.
PATCH_EXAMPLE = '''import numpy as np
from harness.losses import register_loss, sigmoid

@register_loss("per_list_weighted_logloss_v1", kind="pointwise")
def per_list_weighted_logloss(z, y, groups):
    """Pointwise logloss, weighted so every list contributes equally.

    Mechanics only: this is still a pointwise objective and is not expected to
    help. It shows how to register, how to use `groups`, and what to return.
    """
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    weight = np.ones_like(z, dtype=np.float64)
    for g in np.unique(groups):
        m = groups == g
        weight[m] = 1.0 / max(1, m.sum())
    weight /= weight.sum()
    loss = float(-(weight * (y * np.log(p + 1e-9)
                             + (1 - y) * np.log(1 - p + 1e-9))).sum())
    return loss, (weight * (p - y)).astype(np.float32)

CONFIG = {"loss": "per_list_weighted_logloss_v1", "group_by": "user_id"}
'''

#: What CONFIG may contain. Stated because the model cannot see train_fm.
CONFIG_KEYS = """CONFIG is a dict of keyword arguments for the harness trainer.
Every key is optional; an empty CONFIG trains the reference baseline.

  loss      str   the name you registered, or omit for "pointwise_logloss"
  group_by  str   "user_id" or "user_id+date"   -- defines the lists in `groups`
  k         int   embedding dimension           (default 16)
  lr        float learning rate                 (default 0.001)
  l2        float L2 penalty                    (default 1e-6)
  batch     int   batch size                    (default 8192)
  max_epochs int  epoch cap                     (default 40)
  patience  int   early-stop patience on validation primary (default 4)

The trainer handles batching, early stopping on validation primary, restoring the
best epoch, and scoring. You do not write any of that."""


def build_prompt(diagnosis_text: str, *, corpus: str, capabilities: Dict[str, Any],
                 analyses: List[Any] | None = None) -> str:
    """Assemble the proposer prompt from screened parts."""
    sections = [
        '# Task', '',
        'Rank each user\'s logged impressions by likelihood of long_view. '
        'Scored by GAUC and nDCG@5; the primary metric is their mean, computed '
        'within each user\'s own list. The baseline to beat is validation primary '
        '0.6015.', '',
        diagnosis_text, '',
        '# What you can ask the data', '',
        'Call these between iterations to check a belief before spending an '
        'experiment on it:', '',
    ]
    sections += [f'- `{kind}`: {description}'
                 for kind, description in capabilities.get('kinds', {}).items()]
    sections += ['', f"Splits available: {capabilities.get('splits')}. "
                     f"The hidden test split cannot be asked about.", '']

    if analyses:
        sections += ['# Analyses you requested', '']
        for result in analyses:
            sections += [f'### {result.kind} on {result.split} '
                         f'({result.question})', '', result.to_markdown(), '']

    sections += ['# Reference material', '',
                 'Retrieved from the method corpus. This is background, not a queue '
                 'to work through, and every estimate in it should be checked with '
                 'the tools above before it is trusted.', '', corpus, '',
                 '# How a patch is written', '', CONFIG_KEYS, '',
                 'A complete, valid patch looks exactly like this:', '',
                 '```python', PATCH_EXAMPLE, '```', '',
                 'Yours will differ in substance. It must not differ in shape: one '
                 'module, no main(), no argparse, no training loop, ending in '
                 'CONFIG.', '',
                 '# Your task', '',
                 'Propose ONE experiment. State what you believe is wrong and why '
                 'this change should move the metric.', '',
                 '# Output format -- read this last', '',
                 'Return exactly one JSON object and nothing else. No preamble, no '
                 'markdown fence, no commentary after it. The `patch` field holds '
                 'your Python source as a JSON string, with newlines escaped.', '',
                 '{"hypothesis": "<your reasoning, under 120 words>",',
                 ' "target_stage": "objective|model|features|sampling|ensemble",',
                 ' "patch_kind": "<short label>",',
                 ' "expected_gain": <float>,',
                 ' "expected_cost_minutes": <float>,',
                 ' "patch": "<python source>"}', '',
                 'A response that is not a single JSON object is discarded and the '
                 'iteration is wasted.']

    prompt = '\n'.join(sections)
    guards.assert_no_test_metrics(prompt, where='proposer prompt')
    return prompt


def load_corpus(max_chars: int = 14_000) -> str:
    """The method corpus, screened. Reference material, never a queue."""
    path = hdata.repo_root() / 'knowledge' / 'methods.md'
    text = path.read_text(encoding='utf-8') if path.exists() else ''
    guards.assert_no_test_metrics(text, where='method corpus')
    return text[:max_chars]


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def parse(text: str) -> Proposal:
    """Read a proposal out of a model response.

    Tolerant of a fenced code block around the JSON, because that is the commonest
    shape a model returns and rejecting it would waste an iteration on formatting.
    """
    blob = _extract_json(text)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ProposalError(f'the response was not valid JSON: {exc}') from None
    if not isinstance(data, dict):
        raise ProposalError('the response must be a single JSON object')

    missing = [key for key in ('hypothesis', 'target_stage', 'patch') if key not in data]
    if missing:
        raise ProposalError(f'the proposal is missing {missing}')

    return Proposal(
        hypothesis=str(data['hypothesis']).strip(),
        target_stage=str(data['target_stage']).strip(),
        patch_kind=str(data.get('patch_kind', 'unspecified')).strip(),
        patch=str(data['patch']),
        expected_gain=float(data.get('expected_gain', 0.0) or 0.0),
        expected_cost_minutes=float(data.get('expected_cost_minutes', 0.0) or 0.0),
        source='llm',
    )


def _extract_json(text: str) -> str:
    text = (text or '').strip()
    fenced = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end > start:
        return text[start:end + 1]
    raise ProposalError('no JSON object found in the response')


# --------------------------------------------------------------------------
# the proposer
# --------------------------------------------------------------------------

class Proposer:
    """Asks for one experiment, validates it, and repairs it once if it fails."""

    def __init__(self, client: agent_llm.LLMClient, *,
                 corpus: str | None = None,
                 capabilities: Dict[str, Any] | None = None):
        self.client = client
        self.corpus = corpus if corpus is not None else load_corpus()
        self.capabilities = capabilities or hanalyse.capabilities()
        self.last_prompt: str = ''

    def propose(self, diagnosis_text: str, *, tried: List[str] | None = None,
                analyses: List[Any] | None = None,
                role: str = agent_llm.STRONG) -> Proposal:
        """One proposal. Raises ``ProposalError`` if it cannot produce a valid one."""
        if not self.client.enabled:
            raise agent_llm.LLMUnavailable(
                'no provider configured; use deterministic_proposal instead')
        self.last_prompt = build_prompt(diagnosis_text, corpus=self.corpus,
                                        capabilities=self.capabilities,
                                        analyses=analyses)
        response = self.client.complete(self.last_prompt, system=SYSTEM, role=role,
                                        max_tokens=4096)
        return validate(parse(response.text), tried=tried)

    def repair(self, proposal: Proposal, error: str, *,
               tried: List[str] | None = None,
               role: str = agent_llm.FAST) -> Proposal:
        """One repair attempt, on the cheap model. One only, then abandon.

        The traceback goes back to the model with the original patch. This is the
        single retry CLAUDE.md section 6.3 allows; a second would be a search.
        """
        prompt = (
            'Your patch failed. Here is the code and the error.\n\n'
            f'## Your hypothesis\n{proposal.hypothesis}\n\n'
            f'## The patch\n```python\n{proposal.patch}\n```\n\n'
            f'## The failure\n```\n{error[:2000]}\n```\n\n'
            'Fix it and reply with the same JSON object shape, keeping the '
            'hypothesis unless the failure disproves it. If the idea cannot work, '
            'say so in the hypothesis and propose the smallest change that tests '
            'the same belief.')
        guards.assert_no_test_metrics(prompt, where='repair prompt')
        response = self.client.complete(prompt, system=SYSTEM, role=role,
                                        max_tokens=4096)
        repaired = validate(parse(response.text), tried=tried)
        repaired.source = 'llm-repair'
        return repaired


# --------------------------------------------------------------------------
# the deterministic path
# --------------------------------------------------------------------------

#: A fixed sequence over the corpus, for LLM_PROVIDER=none. Insurance against an
#: outage, and the honest control: if this scores as well as the agent, the agent
#: was not adding anything.
#:
#: Deliberately shallow. This is a fallback, not a second agent, and it does not
#: get the good ideas -- writing a pairwise objective is the agent's job.
DETERMINISTIC_SEQUENCE: List[Dict[str, Any]] = [
    {'hypothesis': 'Reproduce the official baseline to confirm the harness, the '
                   'trainer and the scoring path agree with the published number '
                   'before anything is changed.',
     'target_stage': 'model', 'patch_kind': 'baseline_reproduction',
     'config': {}},
    {'hypothesis': 'The baseline stops on validation primary with patience 4. A '
                   'longer patience may find a later peak, since the validation '
                   'curve is noisy at the 0.001 level.',
     'target_stage': 'model', 'patch_kind': 'patience',
     'config': {'patience': 8}},
    {'hypothesis': 'A smaller learning rate may reach a better optimum given that '
                   'the validation curve peaks early and then declines.',
     'target_stage': 'model', 'patch_kind': 'learning_rate',
     'config': {'lr': 0.0005, 'max_epochs': 60, 'patience': 8}},
    {'hypothesis': 'Stronger L2 may delay the overfitting the epoch curve shows.',
     'target_stage': 'model', 'patch_kind': 'regularisation',
     'config': {'l2': 1e-5, 'patience': 8}},
    {'hypothesis': 'A smaller batch gives more updates per epoch at the same data '
                   'cost, which may matter given how early validation peaks.',
     'target_stage': 'sampling', 'patch_kind': 'batch_size',
     'config': {'batch': 4096, 'patience': 8}},
]


#: Applied to the base sequence on each wrap-around, so the fallback does not run
#: dry. Each variant is a genuinely different configuration, not a relabelling.
DETERMINISTIC_VARIANTS: List[Dict[str, Any]] = [
    {},
    {'k': 8},
    {'k': 32},
    {'lr': 0.002},
    {'l2': 1e-4},
    {'batch': 16384},
]


def deterministic_plan() -> List[Dict[str, Any]]:
    """The full fallback plan: the base sequence crossed with the variants.

    Openly a scripted search, which is exactly what the fallback is allowed to be.
    It is not the agent, it is labelled as not the agent in every record it
    produces, and its purpose is to keep producing a valid submission when no model
    is reachable.
    """
    plan: List[Dict[str, Any]] = []
    for variant in DETERMINISTIC_VARIANTS:
        for entry in DETERMINISTIC_SEQUENCE:
            merged = dict(entry)
            merged['config'] = {**entry['config'], **variant}
            if variant:
                merged['patch_kind'] = (f'{entry["patch_kind"]}+'
                                        + '+'.join(f'{k}{v}' for k, v in variant.items()))
            plan.append(merged)
    return plan


def deterministic_proposal(index: int, *,
                           tried: List[str] | None = None) -> Proposal:
    """The *index*-th fallback experiment, skipping anything already tried."""
    tried = tried or []
    plan = deterministic_plan()
    for offset in range(len(plan)):
        entry = plan[(index + offset) % len(plan)]
        patch = (f'"""Deterministic fallback: {entry["patch_kind"]}.\n\n'
                 f'Generated with no LLM call. This is NOT the agent; it is the\n'
                 f'outage fallback and the honest control. Any run using it must\n'
                 f'say so in its report.\n"""\n'
                 f'CONFIG = {entry["config"]!r}\n')
        proposal = Proposal(
            hypothesis=entry['hypothesis'], target_stage=entry['target_stage'],
            patch_kind=entry['patch_kind'], patch=patch, source='deterministic')
        if proposal.content_hash not in tried:
            return validate(proposal, tried=tried)
    raise ProposalError(
        f'all {len(plan)} deterministic fallbacks have been tried. There is '
        f'nothing further this path can propose.')
