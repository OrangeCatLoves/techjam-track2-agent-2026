"""Run logs. One for machines, one for the judges.

OWNS
    - ``log.jsonl``: one screened record per iteration, the machine-readable trail
    - ``log.md``: the same run written for a human, which is what a judge reads
    - the resource report: tokens, wall clock, iterations, interventions
    - recovery events, logged separately from iterations so a restart is visible
      but is not mistaken for an experiment

MUST NEVER
    - write a hidden-test metric anywhere. Every record passes
      ``guards.assert_record_clean`` before it reaches either sink, and the
      markdown is screened again as text before it is written
    - open a file without an explicit encoding. The starter kit is bilingual and
      generated code can print anything; a cp1252 sink turns a successful
      experiment into a crash report (D10)
    - write a number that was not computed by code. The LLM's hypothesis is
      recorded as its hypothesis; its metrics are the harness's

WHY TWO SINKS
    ``log.jsonl`` is for the resource table and for anything programmatic.
    ``log.md`` is a deliverable: it is the evidence that the agent reasoned rather
    than executed a queue, which is what Innovation and Impact are scored on.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from harness import guards

#: Non-experiment events. Logged, counted, and reported separately from
#: iterations, because a restart is not an experiment and an operational recovery
#: is not a manual intervention.
EVENT_RESTART = 'restart'
EVENT_RECOVERY = 'recovery'
EVENT_INTERVENTION = 'manual_intervention'
EVENT_CANARY = 'canary_trip'
EVENT_REVIEW_FLAG = 'review_flag'
EVENT_NOTE = 'note'


class RunLogger:
    """Writes both sinks for one run. Append-only; a restart continues the file."""

    def __init__(self, run_dir: str | Path, run_id: str | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or self.run_dir.name

    # -- paths -------------------------------------------------------------

    @property
    def jsonl_path(self) -> Path:
        return self.run_dir / 'log.jsonl'

    @property
    def markdown_path(self) -> Path:
        return self.run_dir / 'log.md'

    @property
    def events_path(self) -> Path:
        return self.run_dir / 'events.jsonl'

    # -- writing -----------------------------------------------------------

    def _append(self, path: Path, payload: Dict[str, Any], where: str) -> Dict[str, Any]:
        guards.assert_record_clean(payload, where=where)
        with open(path, 'a', encoding='utf-8') as fh:      # explicit: see D10
            fh.write(json.dumps(payload, default=str) + '\n')
        return payload

    def log_iteration(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Record one iteration to both sinks.

        The record shape follows CLAUDE.md section 10.1: hypothesis, target stage,
        patch, metrics, decision and reason, tokens, wall clock, errors.
        """
        payload = dict(record)
        payload.setdefault('run_id', self.run_id)
        payload.setdefault('timestamp', time.strftime('%Y-%m-%dT%H:%M:%S'))
        self._append(self.jsonl_path, payload, where='log.jsonl record')
        self._write_markdown()
        return payload

    def log_event(self, kind: str, message: str, **fields: Any) -> Dict[str, Any]:
        """Record a non-experiment event.

        Restarts and recoveries go here rather than into the iteration log, so the
        resource report can state them separately. Per the organisers' webinar,
        restarting a crashed process is operational recovery, not a manual
        intervention -- but both are counted, and the distinction only survives if
        they are recorded apart.
        """
        payload = {'run_id': self.run_id, 'kind': kind, 'message': message,
                   'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'), **fields}
        return self._append(self.events_path, payload, where='event record')

    # -- reading -----------------------------------------------------------

    def _read(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        out: List[Dict[str, Any]] = []
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def iterations(self) -> List[Dict[str, Any]]:
        return self._read(self.jsonl_path)

    def events(self) -> List[Dict[str, Any]]:
        return self._read(self.events_path)

    # -- the resource report ----------------------------------------------

    def resource_report(self, *, iterations_cap: int = 50,
                        wall_clock_seconds: float | None = None,
                        gpu_hours: float = 0.0) -> Dict[str, Any]:
        """The table CLAUDE.md section 10.2 requires.

        Manual interventions and operational restarts are counted separately and
        labelled, because the definition in force distinguishes them: an
        intervention is a human changing the agent's instructions, objective or
        search space; restarting a crashed process is not.
        """
        records = self.iterations()
        events = self.events()
        tokens_in = sum(r.get('tokens', {}).get('input', 0) or 0 for r in records)
        tokens_out = sum(r.get('tokens', {}).get('output', 0) or 0 for r in records)
        seconds = (wall_clock_seconds if wall_clock_seconds is not None
                   else sum(r.get('wall_clock_s', 0) or 0 for r in records))

        report = {
            'run_id': self.run_id,
            'iterations_used': len(records),
            'iterations_cap': iterations_cap,
            'wall_clock_hours': round(seconds / 3600.0, 3),
            'tokens': {'input': tokens_in, 'output': tokens_out,
                       'total': tokens_in + tokens_out},
            'gpu_hours': gpu_hours,
            'manual_interventions': sum(
                1 for e in events if e.get('kind') == EVENT_INTERVENTION),
            'operational_restarts': sum(
                1 for e in events if e.get('kind') == EVENT_RESTART),
            'recovery_events': sum(
                1 for e in events if e.get('kind') == EVENT_RECOVERY),
            'canary_trips': sum(1 for e in events if e.get('kind') == EVENT_CANARY),
            'review_flags': sum(1 for e in events if e.get('kind') == EVENT_REVIEW_FLAG),
            'intervention_definition': (
                'a human changing the agent instructions, objective or search '
                'space. Restarting a crashed process is operational recovery and '
                'is counted separately.'),
        }
        guards.assert_record_clean(report, where='resource report')
        return report

    # -- the human-readable log -------------------------------------------

    def _write_markdown(self) -> None:
        text = self.markdown()
        guards.assert_no_test_metrics(text, where='log.md')
        self.markdown_path.write_text(text, encoding='utf-8')

    def markdown(self) -> str:
        """The run, written for a person. Regenerated from the JSONL each time.

        Regenerating rather than appending means the file can never drift from the
        machine-readable record, and a restart cannot produce a duplicated tail.
        """
        records = self.iterations()
        events = self.events()
        lines: List[str] = [
            f'# Run `{self.run_id}`', '',
            '_Autonomous ML research agent, TikTok TechJam 2026 Track 2._', '',
            'Every number below was computed by the harness. The hypotheses are the '
            "agent's; the measurements are not.", '',
            'No hidden-test metric appears in this file, or in any log the agent '
            'can read. See CLAUDE.md section 5.', '',
        ]

        if not records:
            lines += ['## Iterations', '', '_No iterations recorded yet._', '']
        else:
            best = max((r for r in records
                        if isinstance(r.get('metrics', {}).get('val_primary'), (int, float))),
                       key=lambda r: r['metrics']['val_primary'], default=None)
            lines += ['## Summary', '',
                      f'- Iterations recorded: **{len(records)}**']
            if best is not None:
                lines.append(f"- Best validation primary: **{best['metrics']['val_primary']:.4f}** "
                             f"(iteration {best.get('iteration', '?')})")
            kept = sum(1 for r in records if r.get('decision') == 'keep')
            lines += [f'- Kept: {kept} | rejected: '
                      f"{sum(1 for r in records if r.get('decision') == 'reject')} | "
                      f"failed: {sum(1 for r in records if r.get('decision') == 'failed')}",
                      '', '## Iterations', '']

            for record in records:
                lines += self._iteration_markdown(record)

        if events:
            lines += ['## Events', '',
                      '| when | kind | detail |', '|---|---|---|']
            for event in events:
                detail = str(event.get('message', '')).replace('|', '\\|')
                lines.append(f"| {event.get('timestamp', '')} | "
                             f"`{event.get('kind', '')}` | {detail} |")
            lines.append('')

        return '\n'.join(lines)

    @staticmethod
    def _iteration_markdown(record: Dict[str, Any]) -> List[str]:
        iteration = record.get('iteration', '?')
        decision = record.get('decision', '?')
        badge = {'keep': 'KEPT', 'reject': 'rejected',
                 'failed': 'FAILED'}.get(decision, decision)
        metrics = record.get('metrics', {}) or {}
        lines = [f'### Iteration {iteration} — {badge}', '']

        if record.get('hypothesis'):
            lines += [f"**Hypothesis.** {record['hypothesis']}", '']
        if record.get('target_stage'):
            lines.append(f"**Target stage.** `{record['target_stage']}`  ")
        if record.get('patch_kind'):
            lines.append(f"**Patch kind.** `{record['patch_kind']}`  ")
        if record.get('target_stage') or record.get('patch_kind'):
            lines.append('')

        if metrics:
            lines += ['| metric | value |', '|---|---|']
            for key in ('val_gauc', 'val_ndcg5', 'val_primary', 'train_primary',
                        'gap', 'mean_list_size'):
                if isinstance(metrics.get(key), (int, float)):
                    lines.append(f'| {key} | {metrics[key]:.4f} |')
            lines.append('')

        if record.get('reason'):
            lines += [f"**Decision.** {badge} — {record['reason']}", '']
        if record.get('errors'):
            errors = record['errors']
            first = errors[0] if isinstance(errors, list) and errors else errors
            lines += ['```', str(first)[:600], '```', '']
        if record.get('flagged_for_review'):
            lines += ['> **Flagged for review.** This result is implausibly good '
                      'for this benchmark and must be inspected by a human before '
                      'submission. See D13.', '']

        cost = []
        if isinstance(record.get('wall_clock_s'), (int, float)):
            cost.append(f"{record['wall_clock_s']:.0f}s")
        tokens = record.get('tokens', {}) or {}
        if tokens.get('input') or tokens.get('output'):
            cost.append(f"{tokens.get('input', 0)} in / {tokens.get('output', 0)} out tokens")
        if cost:
            lines += [f"_Cost: {' · '.join(cost)}._", '']
        return lines


def write_resource_table(report: Dict[str, Any], path: str | Path) -> Path:
    """Write the resource report as a markdown table, for the README and Devpost."""
    tokens = report.get('tokens', {})
    rows = [
        ('Iterations used', f"{report['iterations_used']} of {report['iterations_cap']}"),
        ('Agent wall clock', f"{report['wall_clock_hours']:.2f} h"),
        ('LLM tokens (input + output)', f"{tokens.get('total', 0):,}"),
        ('  input', f"{tokens.get('input', 0):,}"),
        ('  output', f"{tokens.get('output', 0):,}"),
        ('GPU-hours', f"{report.get('gpu_hours', 0)}"),
        ('**Manual interventions**', f"**{report['manual_interventions']}**"),
        ('Operational restarts (not interventions)', str(report['operational_restarts'])),
        ('Recovery events', str(report['recovery_events'])),
        ('Canary trips', str(report.get('canary_trips', 0))),
        ('Results flagged for review', str(report.get('review_flags', 0))),
    ]
    body = '\n'.join(f'| {name} | {value} |' for name, value in rows)
    text = ('## Resource usage\n\n| | |\n|---|---|\n' + body + '\n\n'
            + f"_Manual intervention is {report['intervention_definition']}_\n")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')
    return target
