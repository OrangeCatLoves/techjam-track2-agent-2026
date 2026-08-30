"""The agent loop. Everything else exists so that this can be simple.

OWNS
    - one iteration: diagnose, propose, write, run, decide, log, update the clock
    - the recovery ladder from CLAUDE.md 6.3, and nothing more elaborate than it
    - the stop condition, which is the convergence tracker's answer and nobody
      else's

MUST NEVER
    - stop before convergence fires, or continue after it. Both are enforced by
      ``harness/convergence.py``, and this module asks rather than decides
    - compare a raw score against the best. ``ExperimentResult.usable`` is checked
      first, every time; the tracker refuses a non-finite score precisely because
      a NaN would otherwise reset the strike streak
    - keep a quarantined result, or repair a hard failure. Repairing a canary trip
      or an evaluator rejection is patching around a guard
    - lose state on a crash. Every counter and every artefact is on disk before the
      next iteration starts

THE SHAPE OF ONE ITERATION
    1. diagnose the previous result into facts
    2. propose one experiment (LLM, or the deterministic fallback)
    3. validate and write the patch
    4. run it sandboxed via run_experiment
    5. on a soft failure, one repair attempt; on a hard failure, move on
    6. keep or reject against validation primary, which is the sole authority
    7. log to both sinks, archive the patch, update the tracker
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

from agent import diagnose as agent_diagnose
from agent import llm as agent_llm
from agent import propose as agent_propose
from harness import analyse as hanalyse
from harness import convergence as hconv
from harness import data as hdata
from harness import experiment as hexperiment
from harness import guards
from harness import ledger as hledger
from harness import logger as hlogger


@dataclass
class IterationOutcome:
    """What one turn of the loop did."""
    iteration: int
    decision: str
    val_primary: float | None = None
    error_kind: str | None = None
    repaired: bool = False
    seconds: float = 0.0
    proposal: Dict[str, Any] = field(default_factory=dict)


class AgentLoop:
    """Drives the run. Thin on purpose: the hard parts live in the harness."""

    def __init__(self, *,
                 run_id: str | None = None,
                 run_dir: str | Path | None = None,
                 client: agent_llm.LLMClient | None = None,
                 run_experiment: Callable[..., Any] | None = None,
                 splits: Dict[str, list] | None = None,
                 seed: int = 0,
                 max_epochs: int | None = None):
        self.ledger = hledger.Ledger.open(run_dir, run_id=run_id)
        self.logger = hlogger.RunLogger(self.ledger.run_dir, self.ledger.run_id)
        self.client = client if client is not None else agent_llm.LLMClient(
            on_event=lambda kind, message, fields: self.logger.log_event(
                kind, message, **{k: v for k, v in fields.items()
                                  if k not in ('by_model',)}))
        self.tracker = hconv.ConvergenceTracker.open(self.ledger.convergence_path)
        self.run_experiment = run_experiment or hexperiment.run_experiment
        self.splits = splits
        self.seed = seed
        self.max_epochs = max_epochs
        self.proposer = agent_propose.Proposer(self.client)
        self._last_result: Any = None
        self._last_metrics: Dict[str, float] = {}
        self._analyses: List[Any] | None = None
        self._deterministic_index = 0

    # -- the run -----------------------------------------------------------

    def run(self, *, max_iterations: int | None = None) -> Dict[str, Any]:
        """Iterate until convergence fires. The tracker decides when that is."""
        self.tracker.start_session()
        if self.tracker.iteration:
            self.logger.log_event(
                hlogger.EVENT_RESTART,
                f'resumed at iteration {self.tracker.iteration} with '
                f'{self.tracker.strikes} strike(s); counters were not reset')

        performed = 0
        try:
            while self.tracker.should_continue():
                if max_iterations is not None and performed >= max_iterations:
                    break                     # a development cap, not a stop rule
                self.step()
                performed += 1
        except agent_llm.TokenBudgetExceeded as exc:
            self.logger.log_event('token_ceiling', str(exc))
        finally:
            self.tracker.end_session()

        return self.finish()

    def step(self) -> IterationOutcome:
        """One iteration, start to finish. Never raises for an experiment failure."""
        iteration = self.tracker.iteration + 1
        started = time.time()
        self.ledger.clean_gen()

        proposal, failure = self._get_proposal(iteration)
        if proposal is None:
            return self._record_failure(iteration, 'proposal', failure or 'no proposal',
                                        started, {})

        patch_path = self.ledger.new_patch_path(iteration)
        try:
            hpatch_written = agent_propose.hpatch.write_patch(proposal.patch, patch_path)
        except Exception as exc:
            return self._record_failure(iteration, hexperiment.ERROR_REJECTED,
                                        str(exc), started, proposal.as_record())

        result = self.run_experiment(
            hpatch_written, self.seed,
            checkpoint_path=self.ledger.checkpoint_path(iteration),
            max_epochs=self.max_epochs)
        repaired = False

        if not result.ok and result.error_kind == hexperiment.ERROR_CODE:
            repaired_result = self._attempt_repair(iteration, proposal, result)
            if repaired_result is not None:
                result, repaired = repaired_result, True

        if not result.ok and result.error_kind == hexperiment.ERROR_TIMEOUT:
            self.logger.log_event(hlogger.EVENT_RECOVERY,
                                  'timeout; retrying once at a 30% subsample')
            result = self.run_experiment(
                hpatch_written, self.seed,
                checkpoint_path=self.ledger.checkpoint_path(iteration),
                max_epochs=self.max_epochs, subsample=0.3)

        return self._decide(iteration, proposal, result, started, repaired)

    # -- the pieces --------------------------------------------------------

    def _get_proposal(self, iteration: int):
        """LLM if available, deterministic otherwise. Never both silently."""
        tried = list(self.tracker.tried)
        if not self.client.enabled:
            # The deterministic path can run dry. That is an inability to
            # continue, not a voluntary stop, so it is recorded as a failed
            # iteration and the tracker decides what it means.
            try:
                proposal = agent_propose.deterministic_proposal(
                    self._deterministic_index, tried=tried)
            except agent_propose.ProposalError as exc:
                return None, str(exc)
            self._deterministic_index += 1
            return proposal, None
        try:
            diagnosis = self._diagnose(iteration)
            return self.proposer.propose(diagnosis.to_prompt(), tried=tried,
                                         analyses=self._standing_analyses()), None
        except (agent_propose.ProposalError, agent_llm.LLMError,
                agent_llm.LLMUnavailable) as exc:
            self.logger.log_event(
                hlogger.EVENT_RECOVERY,
                f'proposal failed ({type(exc).__name__}); falling back to the '
                f'deterministic sequence for this iteration')
            try:
                proposal = agent_propose.deterministic_proposal(
                    self._deterministic_index, tried=tried)
                self._deterministic_index += 1
                return proposal, None
            except agent_propose.ProposalError as inner:
                return None, f'{exc} / fallback: {inner}'

    def _diagnose(self, iteration: int) -> agent_diagnose.Diagnosis:
        records = self.ledger.records()
        return agent_diagnose.diagnose(
            self._last_result or hexperiment.ExperimentResult(ok=False,
                                                              error_kind=None),
            iteration=iteration,
            best_primary=(self.ledger.best().val_primary
                          if self.ledger.best() else None),
            previous=self._last_metrics,
            convergence=self.tracker.status(),
            tried=[r.get('patch_kind', '?') for r in records
                   if r.get('patch_kind')],
            # Stages of experiments that were NOT kept, so the diagnosis can tell
            # the agent when it is repeating a direction that has produced nothing.
            stages=[r.get('target_stage') for r in records
                    if r.get('target_stage')
                    and r.get('decision') != hledger.DECISION_KEEP])

    def _standing_analyses(self) -> List[Any]:
        """Measurements put in front of the agent every iteration.

        The agent used these well when they were offered in a one-shot call and
        flew blind without them in the loop, which was a harness gap rather than
        an agent failure: `build_prompt` always accepted analyses and the loop
        never passed any.

        These are primitives, not conclusions. They describe the shape of the data
        and say nothing about what to do with it.
        """
        if self._analyses is None:
            try:
                splits = self.splits if self.splits is not None else hdata.load()
                self._analyses = [
                    hanalyse.analyse('list_size_profile', 'train', splits=splits),
                    hanalyse.analyse('user_composition', 'valid', splits=splits),
                    hanalyse.analyse('cold_key_rate', 'valid', splits=splits),
                ]
            except Exception as exc:
                self.logger.log_event(
                    hlogger.EVENT_RECOVERY,
                    f'standing analyses unavailable: {type(exc).__name__}')
                self._analyses = []
        return self._analyses

    def _attempt_repair(self, iteration: int, proposal, result):
        """One attempt, on the cheap model. Hard failures never reach here."""
        if result.error_kind in hexperiment.HARD_FAILURES:
            return None
        try:
            repaired = self.proposer.repair(proposal, result.error or '',
                                            tried=list(self.tracker.tried))
        except (agent_propose.ProposalError, agent_llm.LLMError,
                agent_llm.LLMUnavailable):
            return None
        self.logger.log_event(hlogger.EVENT_RECOVERY,
                              f'iteration {iteration}: one repair attempt')
        path = agent_propose.hpatch.write_patch(
            repaired.patch, self.ledger.new_patch_path(iteration))
        return self.run_experiment(
            path, self.seed,
            checkpoint_path=self.ledger.checkpoint_path(iteration),
            max_epochs=self.max_epochs)

    def _decide(self, iteration: int, proposal, result, started: float,
                repaired: bool) -> IterationOutcome:
        """Keep or reject. Validation primary is the sole authority."""
        seconds = time.time() - started
        record = proposal.as_record()

        if result.error_kind == hexperiment.ERROR_CANARY:
            self.logger.log_event(
                hlogger.EVENT_CANARY,
                f'iteration {iteration} scored above the leak threshold and was '
                f'quarantined; auditing everything kept so far')
            for flagged in self.ledger.audit():
                self.logger.log_event(
                    hlogger.EVENT_REVIEW_FLAG,
                    f"iteration {flagged.get('iteration')} was kept at "
                    f"{flagged.get('val_primary')} and needs review")

        if not result.usable:
            return self._record_failure(iteration, result.error_kind or 'unknown',
                                        result.error or 'no score', started, record,
                                        repaired=repaired)

        primary = float(result.val_primary)
        improved = self.ledger.would_improve(primary)
        decision = (hledger.DECISION_KEEP if improved else hledger.DECISION_REJECT)
        reason = (f'val_primary {primary:.4f} beats the best'
                  if improved else
                  f'val_primary {primary:.4f} does not beat the best')

        if improved and result.checkpoint:
            self.ledger.promote(iteration, result.checkpoint, primary,
                                metrics={'val_gauc': result.val_gauc,
                                         'val_ndcg5': result.val_ndcg5},
                                flagged_for_review=result.flagged_for_review)
        if result.flagged_for_review:
            self.logger.log_event(
                hlogger.EVENT_REVIEW_FLAG,
                f'iteration {iteration} scored {primary:.4f}, which is implausible '
                f'for this benchmark; kept but must be reviewed before submission')

        self._last_result = result
        self._last_metrics = {'val_gauc': result.val_gauc,
                              'val_ndcg5': result.val_ndcg5,
                              'val_primary': primary}

        status = self.tracker.record_iteration(
            primary, content_hash=proposal.content_hash,
            checkpoint_ref=result.checkpoint)
        self._write(iteration, record, decision, reason, result, seconds, status,
                    repaired)
        self.ledger.archive_patch(iteration, self.ledger.new_patch_path(iteration))
        return IterationOutcome(iteration, decision, primary, None, repaired,
                                seconds, record)

    def _record_failure(self, iteration: int, kind: str, error: str,
                        started: float, record: Dict[str, Any],
                        repaired: bool = False) -> IterationOutcome:
        """An abandoned iteration: one of the 50, but not a strike (Q4)."""
        seconds = time.time() - started
        self._last_result = None
        status = self.tracker.record_failure(
            content_hash=record.get('content_hash'), error=kind)
        self._write(iteration, record, hledger.DECISION_FAILED,
                    f'abandoned after a {kind} failure', None, seconds, status,
                    repaired, errors=[error])
        self.ledger.archive_patch(iteration, self.ledger.new_patch_path(iteration))
        return IterationOutcome(iteration, hledger.DECISION_FAILED, None, kind,
                                repaired, seconds, record)

    def _write(self, iteration: int, record: Dict[str, Any], decision: str,
               reason: str, result, seconds: float, status, repaired: bool,
               errors: List[str] | None = None) -> None:
        """Both sinks and the ledger, in one place so they cannot disagree."""
        metrics: Dict[str, Any] = {}
        if result is not None and result.usable:
            metrics = {'val_gauc': result.val_gauc, 'val_ndcg5': result.val_ndcg5,
                       'val_primary': result.val_primary}
            fit = (result.diagnostics or {}).get('fit') or {}
            for key in ('train_primary', 'gap'):
                if isinstance(fit.get(key), (int, float)):
                    metrics[key] = fit[key]

        payload = {
            'iteration': iteration, 'decision': decision, 'reason': reason,
            'metrics': metrics, 'errors': errors or [],
            'wall_clock_s': round(seconds, 1),
            'strikes_after': status.strikes,
            'repaired': repaired,
            'flagged_for_review': bool(result is not None
                                       and getattr(result, 'flagged_for_review', False)),
            'tokens': {'input': self.client.budget.input_tokens,
                       'output': self.client.budget.output_tokens},
            **record,
        }
        self.logger.log_iteration(payload)
        self.ledger.record({**payload,
                            'val_primary': metrics.get('val_primary')})

    # -- finishing ---------------------------------------------------------

    def write_submission(self) -> Dict[str, Any]:
        """Score the test split with the validation-best checkpoint and write it.

        The scored submission is the **literal validation-best checkpoint**, which
        is why this reads the promoted file rather than whatever the last iteration
        produced. Scoring the test split needs the features and never the label,
        which is why it is possible at all with the label stripped.

        The organisers' own validator is then run over the result. A submission
        that fails ``--check`` is a hard failure: it is reported, never patched
        around.
        """
        from harness import submit as hsubmit
        from harness.models import runners as hrunners

        best = self.ledger.best()
        if best is None:
            return {'written': False, 'reason': 'no checkpoint was ever promoted'}
        try:
            splits = self.splits if self.splits is not None else hdata.load()
            enc, dim = hdata.encode(splits)
            model = hrunners.load_checkpoint(best.checkpoint, dim=dim)
            scores = hrunners.score_split(model, splits, 'test', enc)
            hsubmit.write_split(self.ledger.submission_path, splits, 'test', scores)
            rows = hsubmit.check(self.ledger.submission_path, 'test', splits)
            return {'written': True, 'path': str(self.ledger.submission_path),
                    'rows': len(rows), 'from_iteration': best.iteration,
                    'val_primary': best.val_primary,
                    'flagged_for_review': best.flagged_for_review}
        except Exception as exc:
            self.logger.log_event(
                'submission_failed',
                f'{type(exc).__name__}: {exc}'.replace('\n', ' ')[:300])
            return {'written': False, 'reason': f'{type(exc).__name__}: {exc}'[:300]}

    def finish(self) -> Dict[str, Any]:
        """Write the submission from the validation-best checkpoint, and report."""
        status = self.tracker.status()
        submission = self.write_submission()
        summary = {
            'run_id': self.ledger.run_id,
            'converged': status.converged,
            'reason': status.reason,
            'iterations': status.iteration,
            'best_val_primary': status.best_primary,
            'ledger': self.ledger.summary(),
            'submission': submission,
            'usage': self.client.usage(),
            'resources': self.logger.resource_report(
                wall_clock_seconds=status.elapsed_seconds),
            'review_required': [r for r in self.ledger.audit()],
        }
        guards.assert_record_clean(summary, where='run summary')
        (self.ledger.run_dir / 'summary.json').write_text(
            __import__('json').dumps(summary, indent=2, default=str),
            encoding='utf-8')
        hlogger.write_resource_table(summary['resources'],
                                     self.ledger.run_dir / 'resources.md')
        return summary
