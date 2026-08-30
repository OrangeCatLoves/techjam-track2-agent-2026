"""Facts about the last iteration, computed before the LLM sees anything.

OWNS
    - the rule-based read of what just happened: deltas, which metric moved, the
      overfitting signal, cost, and the state of the run
    - the prompt context the proposer reasons over

MUST NEVER
    - invent a number. Everything here is arithmetic on values the harness
      measured. The LLM hypothesises *on top of* these facts; it does not supply
      them (CLAUDE.md section 6.2)
    - include a hidden-test metric. The diagnosis is screened before it is
      returned, and screened again when it is rendered for a prompt
    - tell the agent what to try. It reports what is true and what has been tried;
      choosing the next move is the proposer's job, and the difference between
      those two things is most of the Innovation score

WHY THE SPLIT MATTERS
    An LLM asked to both measure and decide will confidently mis-measure. Giving it
    arithmetic it cannot get wrong, and then asking only for the judgement, is what
    makes the reasoning in the run log worth reading.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from harness import guards

#: How much a metric must move before it is worth calling a change rather than
#: noise. The organisers' 5-seed std is 0.0008 and adjacent epochs swing ~0.0009,
#: so anything under this is inside the machine's own jitter.
NOISE_FLOOR = 0.001


@dataclass
class Diagnosis:
    """What is true right now. No advice, no predictions."""
    iteration: int
    outcome: str                              # 'improved' | 'no_change' | 'worse' | 'failed'
    facts: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    run_state: Dict[str, Any] = field(default_factory=dict)
    tried: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_prompt(self) -> str:
        """Rendered for the proposer. Screened again on the way out."""
        lines = [f'## Iteration {self.iteration} outcome: {self.outcome}', '']
        if self.facts:
            lines += ['### Measured facts', '']
            lines += [f'- {fact}' for fact in self.facts]
            lines.append('')
        if self.run_state:
            lines += ['### Run state', '']
            lines += [f'- {key}: {value}' for key, value in self.run_state.items()]
            lines.append('')
        if self.tried:
            lines += ['### Already tried (do not repeat)', '']
            lines += [f'- {item}' for item in self.tried]
            lines.append('')
        if self.warnings:
            lines += ['### Warnings', '']
            lines += [f'- {warning}' for warning in self.warnings]
            lines.append('')
        text = '\n'.join(lines)
        guards.assert_no_test_metrics(text, where='diagnosis prompt')
        return text


def _delta_phrase(name: str, current: float, previous: float | None) -> str:
    if previous is None:
        return f'{name} {current:.4f} (first measurement)'
    delta = current - previous
    if abs(delta) < NOISE_FLOOR:
        return (f'{name} {current:.4f}, change {delta:+.4f} '
                f'— inside the {NOISE_FLOOR} noise floor, so not a real move')
    return f'{name} {current:.4f}, change {delta:+.4f}'


def diagnose(result: Any, *, iteration: int, best_primary: float | None,
             previous: Dict[str, float] | None = None,
             convergence: Any = None, tried: List[str] | None = None,
             proposal: Dict[str, Any] | None = None) -> Diagnosis:
    """Turn one experiment result into facts.

    *result* is an ``ExperimentResult``. *previous* is the last iteration's
    metrics, for deltas. *convergence* is a ``ConvergenceStatus``.
    """
    facts: List[str] = []
    warnings: List[str] = []
    metrics: Dict[str, Any] = {}

    if not getattr(result, 'ok', False):
        kind = getattr(result, 'error_kind', None) or 'unknown'
        outcome = 'failed'
        facts.append(f'The experiment did not produce a score. Failure kind: {kind}.')
        error = (getattr(result, 'error', '') or '').strip()
        if error:
            facts.append('Error tail: ' + error.splitlines()[-1][:200])
        if kind in ('evaluator', 'rejected', 'canary'):
            warnings.append(
                f'{kind} is a hard failure: it is never repaired and never retried. '
                f'Propose something different rather than fixing this.')
    elif not getattr(result, 'usable', False):
        outcome = 'failed'
        facts.append('The experiment returned a score that is not a finite number, '
                     'so it cannot be compared against the best. Treated as a '
                     'failure.')
        warnings.append('A non-finite score usually means the objective diverged.')
    else:
        primary = float(result.val_primary)
        metrics = {'val_gauc': result.val_gauc, 'val_ndcg5': result.val_ndcg5,
                   'val_primary': primary}
        gain = None if best_primary is None else primary - best_primary
        if gain is None:
            outcome = 'improved'
            facts.append(f'First scored iteration. val_primary {primary:.4f}.')
        elif gain > NOISE_FLOOR:
            outcome = 'improved'
            facts.append(f'val_primary {primary:.4f}, {gain:+.4f} on the previous '
                         f'best of {best_primary:.4f}.')
        elif abs(gain) <= NOISE_FLOOR:
            outcome = 'no_change'
            facts.append(f'val_primary {primary:.4f}, {gain:+.4f} on the best. '
                         f'That is inside the {NOISE_FLOOR} noise floor, so it is '
                         f'not distinguishable from no change.')
        else:
            outcome = 'worse'
            facts.append(f'val_primary {primary:.4f}, {gain:+.4f} on the best of '
                         f'{best_primary:.4f}.')

        previous = previous or {}
        for key, label in (('val_gauc', 'GAUC'), ('val_ndcg5', 'nDCG@5')):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                facts.append(_delta_phrase(label, float(value), previous.get(key)))

        facts.extend(_diagnostic_facts(getattr(result, 'diagnostics', {}) or {}))

        if getattr(result, 'flagged_for_review', False):
            warnings.append(
                'This score is implausibly high for this benchmark and has been '
                'flagged for human review. It is kept, but treat it with '
                'suspicion rather than as a breakthrough.')

    seconds = getattr(result, 'seconds', None)
    if isinstance(seconds, (int, float)) and seconds:
        facts.append(f'The experiment took {seconds:.0f}s.')

    run_state: Dict[str, Any] = {}
    if convergence is not None:
        run_state = {
            'iteration': f'{convergence.iteration} of {convergence.iteration + convergence.remaining_iterations}',
            'strikes': f'{convergence.strikes} of 3 (three consecutive gains of '
                       f'0.002 or less ends the run)',
            'best_val_primary': ('none yet' if convergence.best_primary is None
                                 else f'{convergence.best_primary:.4f}'),
            'time_remaining_hours': round(convergence.remaining_seconds / 3600, 2),
        }
        # The strike economics, stated from iteration one rather than at the
        # brink. A measured control run converged after FOUR iterations because
        # three consecutive tuning changes each gained under 0.002 -- it used 4 of
        # its 50 iterations and 6 minutes of its 6 hours. An agent that does not
        # know the rule's shape will spend its run the same way.
        #
        # This is teaching the rules of the game, not the answer: which experiment
        # to run is still entirely the agent's to choose.
        if convergence.strikes >= 2:
            warnings.append(
                'One more iteration gaining 0.002 or less ends the run. A '
                'non-improving iteration cannot lower the saved best, so a large '
                'structural change costs nothing that a cautious one preserves.')
        else:
            warnings.append(
                'Three consecutive iterations gaining 0.002 or less end the run, '
                'however many of the 50 remain. Small tuning changes therefore '
                'spend the run quickly: a measured scripted search over '
                'hyperparameters converged after four iterations. A rejected '
                'experiment cannot lower the saved best, so an ambitious change '
                'risks nothing that a cautious one protects.')

    diagnosis = Diagnosis(iteration=iteration, outcome=outcome, facts=facts,
                          metrics=metrics, run_state=run_state,
                          tried=list(tried or []), warnings=warnings)
    guards.assert_record_clean(diagnosis.as_dict(), where='diagnosis')
    return diagnosis


def _diagnostic_facts(diagnostics: Dict[str, Any]) -> List[str]:
    """Read the harness diagnostics into sentences. Arithmetic only."""
    facts: List[str] = []

    fit = diagnostics.get('fit') or {}
    gap, train_primary = fit.get('gap'), fit.get('train_primary')
    if isinstance(gap, (int, float)) and isinstance(train_primary, (int, float)):
        facts.append(f'Train primary {train_primary:.4f} against validation, a gap '
                     f'of {gap:+.4f}.')
    best_epoch, epochs_run = fit.get('best_epoch'), fit.get('epochs_run')
    if isinstance(best_epoch, int) and isinstance(epochs_run, int) and epochs_run:
        if best_epoch < epochs_run:
            facts.append(f'Validation peaked at epoch {best_epoch} of {epochs_run} '
                         f'and then declined.')
        else:
            facts.append(f'Validation was still improving at the last epoch '
                         f'({epochs_run}); training may have stopped early.')

    fields = diagnostics.get('fields') or {}
    if fields:
        ranked = sorted(fields.items(),
                        key=lambda kv: kv[1].get('mean_v_norm', 0.0), reverse=True)
        summary = ', '.join(f'{name} {stats.get("mean_v_norm", 0):.3f}'
                            for name, stats in ranked)
        facts.append(f'Field participation in crosses (mean embedding norm): {summary}.')

    lists = diagnostics.get('lists') or {}
    if lists.get('mean_train_list_size') and lists.get('mean_valid_list_size'):
        facts.append(
            f'Training lists average {lists["mean_train_list_size"]:.1f} rows under '
            f'{lists.get("group_by")}, evaluation lists '
            f'{lists["mean_valid_list_size"]:.1f}.')

    cost = diagnostics.get('cost') or {}
    if cost.get('relative_to_reference'):
        facts.append(f'Cost {cost["relative_to_reference"]:.2f}x a reference FM run '
                     f'({cost.get("reference_fm_seconds")}s).')

    objective = diagnostics.get('objective') or {}
    if objective.get('name'):
        facts.append(f'Objective in use: {objective["name"]} ({objective.get("kind")}).')
    return facts
