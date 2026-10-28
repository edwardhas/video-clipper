"""File-backed ``JsonCache`` for checkpointing paid provider work inside one stage.

The selector computes its own keys (one per candidate window or review batch) and
calls ``put`` as soon as a paid unit finishes, *before* it honours a cancel. Entries
are scoped to the stage's cache key, so a different transcript, model, prompt version
or set of options never sees another run's results. A Resume/Retry of the same stage
then only pays for the units that never completed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..storage import Storage, read_json, stable_hash, write_json_atomic

log = logging.getLogger(__name__)


class FileJsonCache:
    """``JsonCache`` stored at ``data/cache/<namespace>/<sha256(scope_key, key)>.json``."""

    def __init__(self, storage: Storage, namespace: str, scope_key: str) -> None:
        self.storage = storage
        self.namespace = namespace
        self.scope_key = scope_key

    def _path(self, key: str) -> Path:
        return self.storage.cache_path(self.namespace, stable_hash(self.scope_key, key))

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        try:
            return read_json(path)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):     # unreadable/corrupt entry: treat as missing (it is re-requested)
            log.warning("ignoring unreadable %s cache entry", self.namespace)
            return None

    def put(self, key: str, value: Any) -> None:
        write_json_atomic(self._path(key), value)
