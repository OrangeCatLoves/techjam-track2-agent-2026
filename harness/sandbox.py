"""Subprocess isolation for generated code.

OWNS
    - running a Python module in a child process with a wall-clock ceiling and a
      memory ceiling
    - killing that process, and its children, when either ceiling is breached
    - filtering everything the child printed before any of it is returned

MUST NEVER
    - run generated code in this process. An unbounded loop, a memory blow-up or a
      stray ``sys.exit`` in LLM-written code must cost one experiment, not the run
    - return unfiltered child output. Generated code can print anything, including
      a hidden-test metric it computed by accident
    - let a pipe deadlock masquerade as a hang. Child output goes to temporary
      files, not to pipes, because a child that fills a 64 KB pipe buffer while the
      parent is not reading blocks forever and looks exactly like an infinite loop

WINDOWS NOTE
    The POSIX ``resource`` module does not exist here, so ``RLIMIT_AS`` is not
    available. Memory is enforced by polling RSS with ``psutil`` and killing on
    breach. That is a ceiling with a short delay rather than a hard barrier, which
    is adequate: the point is to end a runaway experiment, not to prevent the
    allocation.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import psutil

from harness import guards

#: Polling cadence for the memory and timeout checks.
POLL_SECONDS = 0.25


@dataclass
class SandboxResult:
    """What a sandboxed run is allowed to report."""
    returncode: int
    stdout: str                  # already filtered
    stderr: str                  # already filtered
    seconds: float
    timed_out: bool
    memory_exceeded: bool
    peak_memory_mb: float
    redacted_lines: int
    argv: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (self.returncode == 0 and not self.timed_out
                and not self.memory_exceeded)

    @property
    def failure_kind(self) -> str | None:
        """Maps onto ``error_kind`` in the run_experiment contract."""
        if self.timed_out:
            return 'timeout'
        if self.memory_exceeded:
            return 'memory'
        if self.returncode != 0:
            return 'code'
        return None

    def tail(self, lines: int = 40) -> str:
        """The end of stderr, which is where a traceback lives."""
        return '\n'.join((self.stderr or '').splitlines()[-lines:])


def _process_tree_rss_mb(process: psutil.Process) -> float:
    """Resident memory of a process and its children, in MB."""
    total = 0
    for target in (process, *process.children(recursive=True)):
        try:
            total += target.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / (1024 * 1024)


def _kill_tree(process: psutil.Process) -> None:
    """Kill children first, then the parent, so nothing is reparented and left."""
    for child in process.children(recursive=True):
        try:
            child.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    try:
        process.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


def run_python(args: Sequence[str],
               *,
               timeout_s: float,
               memory_limit_gb: float | None = None,
               cwd: str | os.PathLike | None = None,
               env: dict | None = None) -> SandboxResult:
    """Run ``python <args...>`` in a child process under both ceilings.

    Returns a ``SandboxResult`` in every case. A crash, a timeout and a memory
    breach are all *results*, not exceptions: the agent loop has to record them
    and keep going.
    """
    argv = [sys.executable, *map(str, args)]
    child_env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    if env:
        child_env.update(env)

    limit_mb = None if memory_limit_gb is None else memory_limit_gb * 1024
    started = time.time()
    peak_mb = 0.0
    timed_out = memory_exceeded = False

    # Files rather than pipes: a child that fills a pipe buffer while nobody is
    # reading blocks forever, and is indistinguishable from a hang.
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / 'stdout.txt'
        err_path = Path(tmp) / 'stderr.txt'
        with open(out_path, 'wb') as out_fh, open(err_path, 'wb') as err_fh:
            popen = subprocess.Popen(argv, cwd=None if cwd is None else str(cwd),
                                     env=child_env, stdout=out_fh, stderr=err_fh,
                                     stdin=subprocess.DEVNULL)
            try:
                monitor = psutil.Process(popen.pid)
            except psutil.NoSuchProcess:
                monitor = None

            while True:
                if popen.poll() is not None:
                    break
                elapsed = time.time() - started
                if elapsed > timeout_s:
                    timed_out = True
                    if monitor is not None:
                        _kill_tree(monitor)
                    else:
                        popen.kill()
                    break
                if monitor is not None:
                    try:
                        peak_mb = max(peak_mb, _process_tree_rss_mb(monitor))
                    except psutil.Error:
                        pass
                    if limit_mb is not None and peak_mb > limit_mb:
                        memory_exceeded = True
                        _kill_tree(monitor)
                        break
                time.sleep(POLL_SECONDS)

            try:
                popen.wait(timeout=10)
            except subprocess.TimeoutExpired:
                popen.kill()
                popen.wait(timeout=10)

        seconds = time.time() - started
        raw_out = out_path.read_text(encoding='utf-8', errors='replace')
        raw_err = err_path.read_text(encoding='utf-8', errors='replace')

    clean_out, n_out = guards.filter_stdout(raw_out)
    clean_err, n_err = guards.filter_stdout(raw_err)

    return SandboxResult(
        returncode=popen.returncode if popen.returncode is not None else -1,
        stdout=clean_out, stderr=clean_err, seconds=seconds,
        timed_out=timed_out, memory_exceeded=memory_exceeded,
        peak_memory_mb=peak_mb, redacted_lines=n_out + n_err, argv=argv)
