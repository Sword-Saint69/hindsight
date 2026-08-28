"""Regression tests for issue #3848:

On a large document upsert, delta retain processes thousands of facts at once.
create_temporal_links_batch_per_fact must bound candidate link generation to O(N*K),
using fact_type grouping, date sorting, and a sliding window over the nearest K units.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.retain.link_utils import (
    MAX_TEMPORAL_LINKS_PER_UNIT,
    create_temporal_links_batch_per_fact,
)


@pytest.mark.asyncio
async def test_create_temporal_links_batch_large_delta_is_bounded_on_candidate_generation():
    """Verify candidate link count is structurally bounded by O(N*K) rather than O(N^2)."""
    mock_conn = AsyncMock()
    ops = AsyncMock()
    ops.fetch_temporal_neighbors.return_value = []
    mock_conn.executemany = AsyncMock()
    mock_conn.copy_records_to_table = AsyncMock()

    bank_id = "test_bank"
    now = datetime(2026, 8, 28, 12, 0, 0, tzinfo=UTC)

    # 2,500 valid UUID unit IDs sharing the same fact_type and event_date
    unit_records = [{"id": str(uuid.uuid4()), "event_date": now, "fact_type": "world"} for _ in range(2500)]
    ops.fetch_unit_dates.return_value = unit_records
    unit_ids = [r["id"] for r in unit_records]

    link_count = await create_temporal_links_batch_per_fact(mock_conn, bank_id, unit_ids, ops=ops)

    # Candidate generation is bounded to O(N*K): at most MAX_TEMPORAL_LINKS_PER_UNIT (20) per unit
    max_expected = 2500 * MAX_TEMPORAL_LINKS_PER_UNIT
    assert link_count <= max_expected
    assert link_count > 0


@pytest.mark.asyncio
async def test_create_temporal_links_batch_asymmetric_predecessors_and_close_successor():
    """#3848: Verify that 20 distant predecessors do not suppress a very close successor.

    Unit U has 20 predecessors progressively distant in the past and 1 very close successor
    in the future. Candidate generation must retain the close successor for U.
    """
    mock_conn = AsyncMock()
    ops = AsyncMock()
    ops.fetch_temporal_neighbors.return_value = []

    inserted_links = []

    async def mock_bulk_insert(conn, links, bank_id, skip_exists_check=True, ops=None):
        inserted_links.extend(links)
        return len(links)

    mock_conn.executemany = AsyncMock()
    mock_conn.copy_records_to_table = AsyncMock()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("hindsight_api.engine.retain.link_utils._bulk_insert_links", mock_bulk_insert)

        bank_id = "test_bank"
        target_time = datetime(2026, 8, 28, 12, 0, 0, tzinfo=UTC)
        target_unit_id = str(uuid.uuid4())
        successor_unit_id = str(uuid.uuid4())

        # 20 predecessors from 20 hours ago down to 1 hour ago
        pred_records = [
            {
                "id": str(uuid.uuid4()),
                "event_date": target_time - timedelta(hours=20 - i),
                "fact_type": "world",
            }
            for i in range(20)
        ]
        target_record = {"id": target_unit_id, "event_date": target_time, "fact_type": "world"}
        # 1 very close successor 1 second in the future
        successor_record = {
            "id": successor_unit_id,
            "event_date": target_time + timedelta(seconds=1),
            "fact_type": "world",
        }

        unit_records = pred_records + [target_record, successor_record]
        ops.fetch_unit_dates.return_value = unit_records
        unit_ids = [r["id"] for r in unit_records]

        await create_temporal_links_batch_per_fact(mock_conn, bank_id, unit_ids, ops=ops)

        # Check candidate links originating from target_unit_id
        target_outgoing = [lnk for lnk in inserted_links if lnk[0] == target_unit_id]
        target_outgoing_to_ids = [lnk[1] for lnk in target_outgoing]

        # The very close successor MUST be present in target_unit_id's top links
        assert successor_unit_id in target_outgoing_to_ids, (
            "Close successor was incorrectly suppressed by distant predecessors"
        )
