from bisect import bisect_right
from collections.abc import Sequence


def causal_index(timestamps: Sequence[int], query_timestamp: int) -> int | None:
    """Return the last index at or before a query, never a future sample."""
    index = bisect_right(timestamps, query_timestamp) - 1
    return index if index >= 0 else None


def assert_causal(source_timestamps, availability_timestamps) -> None:
    if any(source > available for source, available in zip(source_timestamps, availability_timestamps)):
        raise ValueError("Future-information leakage detected")
