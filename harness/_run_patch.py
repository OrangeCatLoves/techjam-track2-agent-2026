"""Child-process entry point. Runs one generated patch and writes JSON.

    python -m harness._run_patch <patch_path> <seed> <result_json> [--checkpoint P]

OWNS
    - importing a validated patch, reading its ``CONFIG``, and training with it
    - writing the result, or the failure, as JSON for the parent to read

MUST NEVER
    - be invoked outside ``harness/sandbox.py``. This module has no ceilings of
      its own; the parent owns the timeout and the memory limit
    - print a result. The parent reads the JSON file. Anything printed here is
      incidental, is filtered by the sandbox, and is kept only for the traceback

THE PATCH CONTRACT
    A patch is a Python module that may register objectives with
    ``@harness.losses.register_loss`` and must define::

        CONFIG = {...}   # keyword arguments for harness.models.runners.train_fm

    An empty ``CONFIG`` is legal and trains the reference configuration, which is
    what the agent's own baseline-reproduction iteration does.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _import_patch(path: Path):
    """Import a patch module by path, without putting its directory on sys.path."""
    spec = importlib.util.spec_from_file_location(f'patch_{path.stem}', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot import patch at {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('patch_path')
    parser.add_argument('seed', type=int)
    parser.add_argument('result_json')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--max_epochs', type=int, default=None)
    parser.add_argument('--subsample', type=float, default=None,
                        help='fraction of training rows, for timeout retries')
    args = parser.parse_args(argv)

    payload: Dict[str, Any] = {'ok': False, 'error': None, 'error_kind': None}
    try:
        from harness import data as hdata
        from harness import guards
        from harness.models import runners

        module = _import_patch(Path(args.patch_path))
        config = dict(getattr(module, 'CONFIG', {}) or {})
        if not isinstance(config, dict):
            raise TypeError('a patch CONFIG must be a dict of train_fm keyword '
                            'arguments')
        config.pop('splits', None)
        config.pop('checkpoint_path', None)
        config['seed'] = args.seed
        if args.max_epochs is not None:
            config['max_epochs'] = args.max_epochs

        splits = hdata.load()
        if args.subsample is not None and 0 < args.subsample < 1:
            # Timeout retry path: fewer training rows, evaluation untouched so the
            # score stays comparable to every other iteration.
            keep = int(len(splits['train']) * args.subsample)
            splits = dict(splits, train=splits['train'][:keep])

        result = runners.train_fm(splits, checkpoint_path=args.checkpoint, **config)

        payload = {
            'ok': True,
            'val_gauc': result.val_gauc,
            'val_ndcg5': result.val_ndcg5,
            'val_primary': result.val_primary,
            'diagnostics': result.diagnostics,
            'checkpoint': result.checkpoint,
            'seconds': result.seconds,
            'seed': result.seed,
            'epochs_run': result.epochs_run,
            'best_epoch': result.best_epoch,
            'config': {k: v for k, v in config.items() if isinstance(
                v, (str, int, float, bool, type(None)))},
            'error': None,
            'error_kind': None,
        }
        guards.assert_record_clean(payload, where='patch result')

    except Exception:                       # a failed experiment is data, not a crash
        payload = {'ok': False,
                   'error': traceback.format_exc(limit=12)[-4000:],
                   'error_kind': 'code'}

    Path(args.result_json).write_text(json.dumps(payload, indent=2, default=str),
                                      encoding='utf-8')
    return 0 if payload.get('ok') else 1


if __name__ == '__main__':
    sys.exit(main())
