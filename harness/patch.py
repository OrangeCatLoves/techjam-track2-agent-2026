"""Patch validation. The gate every piece of generated code passes before it runs.

OWNS
    - the protected-path check: a patch that would touch the harness, the tests,
      the config or the organisers' code is rejected outright
    - the import allowlist, enforced by reading the syntax tree rather than the text
    - the dangerous-call check: no ``eval``, no ``exec``, no ``__import__``, no
      filesystem or network access from generated code
    - writing an approved patch into the one directory it is allowed to live in

MUST NEVER
    - be relaxed to make a generated patch pass. If the agent needs a capability
      this forbids, that is a design decision for a human, recorded in
      ``docs/OPEN_QUESTIONS.md``, not a quiet edit here
    - trust a string search. ``import os`` hidden as ``__import__('o' + 's')`` does
      not appear in a grep; it does appear in the AST as a call to ``__import__``

WHAT THIS IS AND IS NOT
    This is a **correctness and scope** gate, not a security sandbox. The threat
    model is a language model writing plausible code that reaches somewhere it
    should not -- reading the raw CSVs directly and so bypassing the test-label
    strip, importing the organisers' ``baseline`` module and calling ``run_fm``
    (which computes a test metric), or editing the convergence rule it is judged
    by. It is not a defence against deliberate sandbox escape, and
    ``harness/sandbox.py`` is what contains the damage when code does run.
"""
from __future__ import annotations

import ast
import fnmatch
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

from harness import data as hdata

#: Modules a generated patch may import, beyond the config allowlist. These are
#: the harness surfaces a patch legitimately needs.
INTERNAL_ALLOWLIST = frozenset({
    'harness.losses',   # to register an objective
    'harness',          # bare `import harness` is harmless on its own
})

#: Names whose mere use is rejected. Each is a way to leave the sandbox's intent.
FORBIDDEN_NAMES = frozenset({
    'eval', 'exec', 'compile', '__import__', 'open', 'input', 'breakpoint',
    'globals', 'locals', 'vars', 'memoryview',
})

#: Modules that are never importable from generated code, even if some future
#: config allowlist edit would otherwise permit them.
NEVER_IMPORTABLE = frozenset({
    'os', 'sys', 'subprocess', 'shutil', 'socket', 'pathlib', 'urllib',
    'requests', 'httpx', 'http', 'ftplib', 'pickle', 'shelve', 'marshal',
    'ctypes', 'multiprocessing', 'threading', 'importlib', 'builtins',
    'starter', 'data', 'baseline', 'evaluate', 'submit',
})


class PatchRejected(ValueError):
    """A patch failed validation. It is never repaired here; it is rejected."""


@dataclass
class PatchReport:
    """The outcome of validating one patch."""
    ok: bool
    path: str
    reasons: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)

    def raise_if_rejected(self) -> 'PatchReport':
        if not self.ok:
            raise PatchRejected(
                f'patch {self.path} rejected:\n  - ' + '\n  - '.join(self.reasons))
        return self


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def _agent_config() -> dict:
    return hdata.load_config().get('agent', {}) or {}


def protected_paths() -> tuple:
    """Paths generated code may never touch. From ``configs/base.yaml``."""
    return tuple(_agent_config().get('protected_paths', ()))


def writable_paths() -> tuple:
    """The only paths generated code may be written to."""
    return tuple(_agent_config().get('writable_paths', ()))


def import_allowlist() -> frozenset:
    """Third-party roots a patch may import, plus the internal surfaces."""
    return frozenset(_agent_config().get('import_allowlist', ())) | INTERNAL_ALLOWLIST


def _relative(path: str | Path) -> str:
    """Repo-relative POSIX path, for matching against the config lists."""
    target = Path(path)
    root = hdata.repo_root()
    try:
        rel = target.resolve().relative_to(root.resolve())
    except ValueError:
        rel = target
    return rel.as_posix()


def is_protected(path: str | Path) -> bool:
    """True if *path* is on the protected list.

    A list entry ending in ``/`` protects a whole tree; anything else is an exact
    file match. Glob patterns are honoured so the list can grow without code
    changes.
    """
    rel = _relative(path)
    for entry in protected_paths():
        entry = str(entry).strip()
        if not entry:
            continue
        if entry.endswith('/'):
            if rel == entry.rstrip('/') or rel.startswith(entry):
                return True
        elif rel == entry or fnmatch.fnmatch(rel, entry):
            return True
    return False


def is_writable(path: str | Path) -> bool:
    """True if *path* sits inside a directory generated code may write to."""
    rel = _relative(path)
    return any(rel.startswith(str(entry).strip())
               for entry in writable_paths() if str(entry).strip())


def check_paths(paths: Iterable[str | Path]) -> List[str]:
    """Reasons why any of *paths* may not be written by generated code."""
    reasons: List[str] = []
    for path in paths:
        rel = _relative(path)
        if is_protected(path):
            reasons.append(
                f'{rel} is a protected path; generated code may never modify it '
                f'(configs/base.yaml agent.protected_paths)')
        elif not is_writable(path):
            reasons.append(
                f'{rel} is outside the writable paths {writable_paths()}')
    return reasons


# --------------------------------------------------------------------------
# source validation
# --------------------------------------------------------------------------

def _import_roots(tree: ast.AST) -> List[str]:
    """Every module a source file imports, as dotted names."""
    found: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:                      # a relative import
                found.append('.' * node.level + (node.module or ''))
            elif node.module:
                found.append(node.module)
    return found


def _import_reasons(imports: Sequence[str]) -> List[str]:
    allowed = import_allowlist()
    reasons: List[str] = []
    for name in imports:
        if name.startswith('.'):
            reasons.append(f'relative import {name!r} is not allowed in a patch')
            continue
        root = name.split('.')[0]
        if root in NEVER_IMPORTABLE:
            reasons.append(
                f'import of {name!r} is never permitted from generated code'
                + ('; the organisers\' modules are reached through the harness, '
                   'and starter.baseline.run_fm computes a hidden-test metric'
                   if root in {'starter', 'data', 'baseline', 'evaluate', 'submit'}
                   else ''))
        elif name not in allowed and root not in allowed:
            reasons.append(
                f'import of {name!r} is not on the allowlist {sorted(allowed)}')
    return reasons


def _name_reasons(tree: ast.AST) -> List[str]:
    reasons: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            reasons.append(f'use of {node.id!r} is not permitted in a patch')
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            reasons.append(f'attribute access to {node.attr!r} is not permitted')
        elif isinstance(node, ast.Name) and node.id.startswith('__') \
                and node.id.endswith('__') and node.id != '__name__':
            reasons.append(f'use of dunder {node.id!r} is not permitted in a patch')
    return sorted(set(reasons))


def validate_source(source: str, path: str | Path = '<patch>') -> PatchReport:
    """Validate patch *source* without writing it anywhere."""
    rel = _relative(path)
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return PatchReport(ok=False, path=rel,
                           reasons=[f'syntax error at line {exc.lineno}: {exc.msg}'])

    imports = _import_roots(tree)
    reasons = _import_reasons(imports) + _name_reasons(tree)
    reasons += check_paths([path])
    return PatchReport(ok=not reasons, path=rel, reasons=reasons,
                       imports=sorted(set(imports)))


def validate_patch(path: str | Path) -> PatchReport:
    """Validate a patch file already on disk."""
    target = Path(path)
    if not target.exists():
        return PatchReport(ok=False, path=_relative(target),
                           reasons=[f'no such patch file: {target}'])
    return validate_source(target.read_text(encoding='utf-8'), target)


def content_hash(source_or_path: str | Path) -> str:
    """Stable short hash of patch content, for the tried-set.

    Hashes the *content*, not the filename, so the same experiment proposed twice
    under different names is recognised as already tried. Whitespace-insensitive
    at the line level, so a reformat is not a new experiment.

    This is what a canary trip records: the exact patch that leaked must never be
    re-proposed, even though the leak class may well be re-encountered.
    """
    candidate = Path(source_or_path) if isinstance(source_or_path, Path) else None
    if candidate is None and isinstance(source_or_path, str) \
            and source_or_path.endswith('.py'):
        maybe = Path(source_or_path)
        candidate = maybe if maybe.exists() else None
    text = (candidate.read_text(encoding='utf-8') if candidate is not None
            else str(source_or_path))
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return hashlib.sha256('\n'.join(lines).encode('utf-8')).hexdigest()[:16]


def write_patch(source: str, path: str | Path) -> Path:
    """Validate *source*, then write it. Rejected patches never reach disk.

    Writing after validating is deliberate: a rejected patch that had already been
    written could be imported by a later run that skipped the check.
    """
    target = Path(path)
    validate_source(source, target).raise_if_rejected()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding='utf-8')
    return target
