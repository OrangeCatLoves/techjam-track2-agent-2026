"""The agent half: the meter, the diagnosis, the proposer, and the loop.

Everything here runs against the stub or a fake transport. No network, no API key,
no training. That is the point of the M2a/M2b split: when the two halves join, only
the seam is new.

The load-bearing assertions:

* the token meter charges **failed** calls, because a counter that increments only
  on success reports a Feasibility number wrong by an unknown amount
* the loop never dies on an experiment failure, and every failure kind in the
  contract is exercised in one run
* a NaN is recorded as a failure, not as a keep and not as a strike-clearing
  improvement
"""
from __future__ import annotations

import json

import pytest

from agent import diagnose as agent_diagnose
from agent import llm as agent_llm
from agent import loop as agent_loop
from agent import propose as agent_propose
from harness import experiment as X
from harness import guards


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class FakeUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeBlock:
    def __init__(self, text):
        self.text = text


class FakeMessage:
    def __init__(self, text, input_tokens=100, output_tokens=50, model='fake-model'):
        self.content = [FakeBlock(text)]
        self.usage = FakeUsage(input_tokens, output_tokens)
        self.model = model
        self.stop_reason = 'end_turn'


def transport_returning(text, **usage):
    def send(**kwargs):
        return FakeMessage(text, model=kwargs['model'], **usage)
    return send


def transport_raising(exc=RuntimeError('upstream is down')):
    def send(**kwargs):
        raise exc
    return send


def client(transport, **kwargs):
    return agent_llm.LLMClient(
        provider='anthropic',
        models={'fast': 'fake-fast', 'strong': 'fake-strong'},
        transport=transport, backoff_seconds=0.0, **kwargs)


VALID_PROPOSAL = json.dumps({
    'hypothesis': 'Pointwise logloss is misaligned with the within-user ranking '
                  'metrics, so a change to the objective should move GAUC.',
    'target_stage': 'objective', 'patch_kind': 'new_loss_function',
    'expected_gain': 0.01, 'expected_cost_minutes': 2.0,
    'patch': 'CONFIG = {"max_epochs": 2}\n'})


# --------------------------------------------------------------------------
# the token meter
# --------------------------------------------------------------------------

def test_a_successful_call_is_charged():
    c = client(transport_returning('hello', input_tokens=120, output_tokens=30))
    response = c.complete('a prompt')
    assert response.ok and not response.estimated
    assert c.budget.input_tokens == 120 and c.budget.output_tokens == 30
    assert c.budget.calls == 1 and c.budget.failed_calls == 0


def test_failed_calls_are_charged_too():
    """The requirement that is expensive to retrofit.

    Failed requests, retries and timeouts consume tokens. A meter that increments
    only on success reports a Feasibility number wrong by an unknown amount, and
    Feasibility is 15% of the grade.
    """
    c = client(transport_raising(), max_retries=3)
    with pytest.raises(agent_llm.LLMError):
        c.complete('a prompt that will fail')
    assert c.budget.calls == 3, 'every attempt is a call'
    assert c.budget.failed_calls == 3
    assert c.budget.spent > 0, 'a failed request still consumed input tokens'
    assert c.budget.estimated_calls == 3


def test_estimated_tokens_are_reported_as_estimated():
    c = client(transport_raising(), max_retries=1)
    with pytest.raises(agent_llm.LLMError):
        c.complete('x' * 4000)
    usage = c.usage()
    assert usage['estimated_calls'] == 1
    assert 'estimated' in usage['note']
    assert usage['total'] >= 1000, 'a 4000-character prompt is not free'


def test_a_retry_that_eventually_succeeds_charges_both_attempts():
    state = {'calls': 0}

    def flaky(**kwargs):
        state['calls'] += 1
        if state['calls'] == 1:
            raise RuntimeError('transient')
        return FakeMessage('ok', input_tokens=200, output_tokens=40,
                           model=kwargs['model'])

    c = client(flaky, max_retries=3)
    response = c.complete('prompt')
    assert response.ok and response.attempts == 2
    assert c.budget.calls == 2 and c.budget.failed_calls == 1
    assert c.budget.input_tokens > 200, 'the failed attempt was charged as well'


def test_the_ceiling_stops_the_run_before_spending():
    c = client(transport_returning('ok'), budget=agent_llm.TokenBudget(limit=100))
    c.budget.input_tokens = 100
    with pytest.raises(agent_llm.TokenBudgetExceeded):
        c.complete('prompt')
    assert c.budget.calls == 0, 'the ceiling is checked before the call, not after'


def test_the_warning_fires_once():
    budget = agent_llm.TokenBudget(limit=10_000, warn_at=100)
    events = []
    c = client(transport_returning('ok', input_tokens=200, output_tokens=10),
               budget=budget,
               on_event=lambda kind, message, fields: events.append(kind))
    c.complete('a')
    c.complete('b')
    assert events.count('token_warning') == 1


def test_model_attribution_is_recorded_per_call():
    """Both roles point at the same model today. Recording it now means the
    fast/strong split is later a config change, not an instrumentation project."""
    c = client(transport_returning('ok'))
    c.complete('a', role=agent_llm.FAST)
    c.complete('b', role=agent_llm.STRONG)
    assert set(c.budget.by_model) == {'fake-fast', 'fake-strong'}
    assert [call.role for call in c.calls] == ['fast', 'strong']


def test_a_prompt_carrying_a_test_metric_is_refused():
    """The last point at which this text is still ours."""
    c = client(transport_returning('ok'))
    with pytest.raises(guards.LeakageError):
        c.complete('here is the result: test GAUC 0.6610 primary 0.5946')
    assert c.budget.calls == 0


def test_deterministic_mode_makes_no_calls():
    c = agent_llm.LLMClient(provider='none')
    assert not c.enabled
    with pytest.raises(agent_llm.LLMUnavailable):
        c.complete('anything')
    assert c.budget.spent == 0


def test_usage_is_screened():
    c = client(transport_returning('ok'))
    c.complete('a')
    guards.assert_record_clean(c.usage(), where='usage')


# --------------------------------------------------------------------------
# the diagnosis
# --------------------------------------------------------------------------

def scored(primary, **extra):
    return X.ExperimentResult(ok=True, val_gauc=primary + 0.03,
                              val_ndcg5=primary - 0.03, val_primary=primary,
                              seconds=63.0, **extra)


def test_a_real_gain_is_reported_as_improved():
    d = agent_diagnose.diagnose(scored(0.6200), iteration=2, best_primary=0.6015)
    assert d.outcome == 'improved'
    assert any('+0.0185' in fact for fact in d.facts)


def test_a_gain_inside_the_noise_floor_is_not_a_change():
    """0.0005 is smaller than the machine's own jitter between adjacent epochs."""
    d = agent_diagnose.diagnose(scored(0.6020), iteration=2, best_primary=0.6015)
    assert d.outcome == 'no_change'
    assert any('noise floor' in fact for fact in d.facts)


def test_a_worse_score_is_reported_as_worse():
    d = agent_diagnose.diagnose(scored(0.5900), iteration=2, best_primary=0.6015)
    assert d.outcome == 'worse'


def test_a_nan_score_is_diagnosed_as_a_failure():
    d = agent_diagnose.diagnose(scored(float('nan')), iteration=2,
                                best_primary=0.6015)
    assert d.outcome == 'failed'
    assert any('not a finite number' in fact for fact in d.facts)


@pytest.mark.parametrize('kind', ['evaluator', 'rejected', 'canary'])
def test_hard_failures_are_flagged_as_not_repairable(kind):
    result = X.ExperimentResult(ok=False, error_kind=kind, error='boom')
    d = agent_diagnose.diagnose(result, iteration=3, best_primary=0.6015)
    assert d.outcome == 'failed'
    assert any('hard failure' in w for w in d.warnings)


def test_the_agent_is_told_when_one_strike_remains():
    class Status:
        iteration, strikes, remaining_iterations = 7, 2, 43
        best_primary, remaining_seconds = 0.6120, 7200.0

    d = agent_diagnose.diagnose(scored(0.6125), iteration=8, best_primary=0.6120,
                                convergence=Status())
    assert any('ends the run' in w for w in d.warnings)
    assert any('costs nothing' in w for w in d.warnings), (
        'and that a big swing is therefore cheap, which is the point')


def test_a_flagged_result_is_described_with_suspicion():
    d = agent_diagnose.diagnose(scored(0.7210, flagged_for_review=True),
                                iteration=4, best_primary=0.6015)
    assert any('suspicion' in w for w in d.warnings)


def test_the_diagnosis_prompt_is_screened():
    d = agent_diagnose.diagnose(scored(0.6200), iteration=2, best_primary=0.6015)
    guards.assert_no_test_metrics(d.to_prompt(), where='diagnosis')
    guards.assert_record_clean(d.as_dict(), where='diagnosis')


def test_diagnostics_are_turned_into_sentences_not_invented():
    result = scored(0.6200, diagnostics={
        'fit': {'train_primary': 0.6620, 'gap': 0.0420, 'best_epoch': 7,
                'epochs_run': 11},
        'lists': {'group_by': 'user_id', 'mean_train_list_size': 43.5,
                  'mean_valid_list_size': 5.6},
        'cost': {'relative_to_reference': 1.02, 'reference_fm_seconds': 63.0}})
    d = agent_diagnose.diagnose(result, iteration=2, best_primary=0.6015)
    joined = ' '.join(d.facts)
    assert 'peaked at epoch 7 of 11' in joined
    assert '43.5' in joined and '5.6' in joined


# --------------------------------------------------------------------------
# the proposer
# --------------------------------------------------------------------------

def test_a_well_formed_proposal_parses():
    proposal = agent_propose.parse(VALID_PROPOSAL)
    assert proposal.target_stage == 'objective'
    assert proposal.source == 'llm'
    agent_propose.validate(proposal)


def test_a_fenced_json_block_parses():
    """The commonest shape a model returns. Rejecting it would waste an iteration
    on formatting."""
    assert agent_propose.parse(f'Here you go:\n```json\n{VALID_PROPOSAL}\n```').patch


def test_prose_without_json_is_rejected():
    with pytest.raises(agent_propose.ProposalError):
        agent_propose.parse('I think we should try a pairwise loss.')


@pytest.mark.parametrize('mutation,reason', [
    ({'hypothesis': 'try stuff'}, 'hypothesis too short'),
    ({'target_stage': 'vibes'}, 'unknown stage'),
    ({'patch': 'x = 1\n'}, 'no CONFIG'),
    ({'patch': 'import os\nCONFIG = {}\n'}, 'forbidden import'),
    ({'patch': 'from baseline import run_fm\nCONFIG = {}\n'}, 'reaches the organisers'),
    ({'patch': 'CONFIG = {}\n' + 'x = 1\n' * 9000}, 'oversized'),
])
def test_malformed_proposals_are_rejected(mutation, reason):
    data = json.loads(VALID_PROPOSAL)
    data.update(mutation)
    with pytest.raises(agent_propose.ProposalError):
        agent_propose.validate(agent_propose.parse(json.dumps(data)))


def test_an_already_tried_experiment_is_rejected():
    proposal = agent_propose.parse(VALID_PROPOSAL)
    with pytest.raises(agent_propose.ProposalError, match='already been tried'):
        agent_propose.validate(proposal, tried=[proposal.content_hash])


def test_the_record_view_omits_the_patch_body():
    """The patch is archived as a file; the log carries its hash and size."""
    record = agent_propose.parse(VALID_PROPOSAL).as_record()
    assert 'patch' not in record
    assert record['content_hash'] and record['patch_chars'] > 0


def test_the_deterministic_plan_is_large_enough_to_not_run_dry():
    plan = agent_propose.deterministic_plan()
    assert len(plan) >= 25, 'the run may reach 50 iterations'
    hashes = {agent_propose.deterministic_proposal(i).content_hash
              for i in range(len(plan))}
    assert len(hashes) == len(plan), 'every fallback must be a distinct experiment'


def test_deterministic_proposals_are_valid_and_labelled():
    proposal = agent_propose.deterministic_proposal(0)
    agent_propose.validate(proposal)
    assert proposal.source == 'deterministic'
    assert 'NOT the agent' in proposal.patch, (
        'a reader must never mistake the fallback for the agent')


def test_deterministic_proposals_skip_what_was_tried():
    first = agent_propose.deterministic_proposal(0)
    second = agent_propose.deterministic_proposal(0, tried=[first.content_hash])
    assert second.content_hash != first.content_hash


def test_the_proposer_prompt_carries_the_corpus_and_the_tools():
    prompt = agent_propose.build_prompt(
        'diagnosis text', corpus='corpus text',
        capabilities={'kinds': {'list_size_profile': 'impressions per user'},
                      'splits': ['train', 'valid']})
    assert 'corpus text' in prompt and 'list_size_profile' in prompt
    assert 'not a queue' in prompt, 'the corpus is reference material, not a plan'
    guards.assert_no_test_metrics(prompt, where='prompt')


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

def build_loop(tmp_path, sequence, **kwargs):
    return agent_loop.AgentLoop(
        run_dir=tmp_path / 'run', client=agent_llm.LLMClient(provider='none'),
        run_experiment=X.StubRunner(sequence), **kwargs)


ALL_FAILURE_KINDS = ['improvement', 'code_error', 'timeout', 'memory_error',
                     'evaluator_rejection', 'nan_score', 'canary_trip',
                     'regression', 'no_improvement', 'no_improvement',
                     'no_improvement']


def test_the_loop_survives_every_failure_kind_in_the_contract(tmp_path):
    """One run, every row of the stub table. Nothing raises."""
    loop = build_loop(tmp_path, ALL_FAILURE_KINDS)
    summary = loop.run(max_iterations=11)
    assert summary['iterations'] >= 5
    ledger = summary['ledger']
    assert ledger['failed'] >= 4, 'the failure kinds must be recorded as failures'
    assert ledger['kept'] >= 1


def test_a_nan_is_recorded_as_a_failure_not_a_keep(tmp_path):
    loop = build_loop(tmp_path, ['improvement', 'nan_score'])
    loop.run(max_iterations=2)
    records = loop.ledger.records()
    assert records[1]['decision'] == 'failed'
    assert loop.tracker.failed_iterations == 1
    assert loop.tracker.strikes == 0, 'a failure leaves the streak untouched'


def test_a_canary_trip_triggers_a_retrospective_audit(tmp_path):
    loop = build_loop(tmp_path, ['improvement', 'canary_trip'])
    loop.run(max_iterations=2)
    kinds = [e['kind'] for e in loop.logger.events()]
    assert 'canary_trip' in kinds
    assert loop.ledger.records()[1]['decision'] == 'failed'


def test_the_run_does_not_halt_on_a_canary_trip(tmp_path):
    """D13: the run continues. Halting an unattended run submits nothing."""
    loop = build_loop(tmp_path, ['canary_trip', 'canary_trip', 'improvement'])
    loop.run(max_iterations=3)
    assert loop.tracker.iteration == 3


def test_convergence_stops_the_loop(tmp_path):
    loop = build_loop(tmp_path, ['improvement', 'no_improvement', 'no_improvement',
                                 'no_improvement', 'improvement'])
    summary = loop.run(max_iterations=10)
    assert summary['converged']
    assert summary['reason'] == 'no_improvement'
    assert summary['iterations'] == 4, 'it must stop the moment the rule fires'


def test_the_loop_writes_every_artefact(tmp_path):
    loop = build_loop(tmp_path, ['improvement'])
    loop.run(max_iterations=1)
    for name in ('ledger.jsonl', 'log.jsonl', 'log.md', 'convergence.json',
                 'summary.json', 'resources.md'):
        assert (loop.ledger.run_dir / name).exists(), name
    assert loop.ledger.best() is not None


def test_a_kept_patch_is_archived_out_of_the_generated_directory(tmp_path):
    loop = build_loop(tmp_path, ['improvement'])
    loop.run(max_iterations=1)
    assert (loop.ledger.run_dir / 'patches' / 'iter_001.py').exists()
    assert not list(loop.ledger.gen_dir().glob('iter_*.py'))


def test_deterministic_mode_spends_no_tokens(tmp_path):
    loop = build_loop(tmp_path, ['improvement', 'no_improvement'])
    summary = loop.run(max_iterations=2)
    assert summary['usage']['total'] == 0
    assert summary['usage']['provider'] == 'none'


def test_a_restart_resumes_rather_than_restarts(tmp_path):
    first = build_loop(tmp_path, ['improvement', 'improvement'])
    first.run(max_iterations=2)
    assert first.tracker.iteration == 2

    resumed = build_loop(tmp_path, ['no_improvement'])
    assert resumed.tracker.iteration == 2, 'counters resume from disk'
    assert resumed.ledger.best() is not None, 'and so does the winner'
    resumed.run(max_iterations=1)
    assert resumed.tracker.iteration == 3
    assert any(e['kind'] == 'restart' for e in resumed.logger.events())


def test_the_summary_is_screened_and_reports_resources(tmp_path):
    loop = build_loop(tmp_path, ['improvement', 'no_improvement'])
    summary = loop.run(max_iterations=2)
    guards.assert_record_clean(summary, where='summary')
    assert summary['resources']['manual_interventions'] == 0
    assert summary['resources']['iterations_used'] >= 1
    assert 'review_required' in summary


# --------------------------------------------------------------------------
# the submission, from the validation-best checkpoint
# --------------------------------------------------------------------------

def test_no_checkpoint_means_no_submission(tmp_path):
    """Reported, not crashed. A run that never kept anything has nothing to send."""
    loop = build_loop(tmp_path, ['code_error'])
    summary = loop.run(max_iterations=1)
    assert summary['submission']['written'] is False
    assert 'no checkpoint' in summary['submission']['reason']


def test_a_broken_checkpoint_is_reported_not_raised(tmp_path):
    """The stub writes a 2x2 checkpoint that cannot score the real encoding.

    That is a useful accident: it exercises the path where submission writing
    fails, which must be a reported outcome rather than an exception that loses
    the whole run's log.
    """
    loop = build_loop(tmp_path, ['improvement'])
    summary = loop.run(max_iterations=1)
    assert summary['submission']['written'] in (True, False)
    if not summary['submission']['written']:
        assert 'reason' in summary['submission']
        assert any(e['kind'] == 'submission_failed' for e in loop.logger.events())


@pytest.mark.slow
@pytest.mark.data
def test_the_full_stack_produces_a_valid_submission(tmp_path):
    """Real training, real checkpoint, real 170,588-row submission.

    This is the M2 gate in miniature: the loop runs unattended, keeps the best
    checkpoint, scores the hidden split with it, and the organisers' own validator
    accepts the result.
    """
    loop = agent_loop.AgentLoop(
        run_dir=tmp_path / 'real',
        client=agent_llm.LLMClient(provider='none'),
        max_epochs=1)
    summary = loop.run(max_iterations=2)

    submission = summary['submission']
    assert submission['written'] is True, submission.get('reason')
    assert submission['rows'] == 170588
    assert submission['from_iteration'] == loop.ledger.best().iteration
    assert loop.ledger.submission_path.exists()

    header = loop.ledger.submission_path.read_text(
        encoding='utf-8').splitlines()[0]
    assert header == 'row_id,user_id,video_id,score'
    guards.assert_record_clean(summary, where='real run summary')
