"""Offline-first operation (shortcoming #14, critical rule #6).

Rural Tanzanian districts have intermittent connectivity, and those are exactly
the districts with the highest disease burden. Offline capability is therefore a
first-class feature, not a degraded mode:

* `local_cache`  — a SQLite store of predictions, alerts and driver data that
  serves the API and dashboard with no network at all;
* `sync_manager` — reconciles the local node with the central instance when a
  link appears, and flushes queued alerts;
* `lightweight_model` — a compact distilled model that runs a forecast on a
  low-spec edge device without the full feature pipeline.
"""

from offline.local_cache import LocalCache  # noqa: F401
from offline.lightweight_model import LightweightModel  # noqa: F401
from offline.sync_manager import SyncManager  # noqa: F401
