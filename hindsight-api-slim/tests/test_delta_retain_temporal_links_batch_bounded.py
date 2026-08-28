"""Regression test for issue #3848:

On a large document upsert, delta retain processes thousands of facts at once.
create_temporal_links_batch_per_fact must bound memory and complete within
milliseconds, using fact_type grouping, date sorting, sliding window, and inline
per-unit link caps rather than building an O(N^2) link list in Python memory.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.retain.link_utils import (
    MAX_TEMPORAL_LINKS_PER_UNIT,
    create_temporal_links_batch_per_fact,
)


@pytest.mark.asyncio
async def test_create_temporal_links_batch_large_delta_is_bounded_and_fast():
    """Verify that 2,500 same-day, same-fact_type units do not trigger O(N^2) memory blowout."""
    mock_conn = AsyncMock()
    ops = AsyncMock()
    # Mock LATERAL DB neighbor fetch returning empty (simulating no DB neighbors)
    ops.fetch_temporal_neighbors.return_value = []
    # Mock bulk link insertion
    mock_conn.executemany = AsyncMock()
    mock_conn.copy_records_to_table = AsyncMock()

    bank_id = "test_bank"
    now = datetime(2026, 8, 28, 12, 0, 0, tzinfo=UTC)

    # 2,500 valid UUID unit IDs sharing the same fact_type and event_date
    import uuid
    unit_records = [
        {"id": str(uuid.uuid4()), "event_date": now, "fact_type": "world"}
        for _ in range(2500)
    ]
    ops.fetch_unit_dates.return_value = unit_records
    unit_ids = [r["id"] for r in unit_records]

    t0 = time.perf_counter()
    link_count = await create_temporal_links_batch_per_fact(
        mock_conn, bank_id, unit_ids, ops=ops
    )
    duration = time.perf_counter() - t0

    # Must complete in under 0.5s (previously would take 5-10 seconds and gigabytes of RAM)
    assert duration < 0.5, f"Temporal link creation took too long: {duration:.3f}s"

    # Link count per unit is bounded by MAX_TEMPORAL_LINKS_PER_UNIT (20) * 2500 units
    max_expected = 2500 * MAX_TEMPORAL_LINKS_PER_UNIT
    assert link_count <= max_expected
    assert link_count > 0
