"""Write one JSON object per prediction request using standard logging."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


@contextmanager
def jsonl_logger(path: Path) -> Iterator[logging.Logger]:
    """Keep each app's handler separate and close its file on shutdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"bike_demand.requests.{uuid4().hex}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    try:
        yield logger
    finally:
        logger.removeHandler(handler)
        handler.close()


def request_details(body: object) -> dict[str, object]:
    """Extract input summaries even when the request fails validation."""
    if not isinstance(body, dict):
        return {}
    target_hour = body.get("target_hour")
    counts = body.get("recent_counts")
    return {
        "target_hour": target_hour if isinstance(target_hour, str) else None,
        "history_hours": len(counts) if isinstance(counts, list) else None,
        "missing_count": (
            sum(value is None for value in counts) if isinstance(counts, list) else None
        ),
    }
