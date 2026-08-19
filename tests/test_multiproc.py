"""Tests for tstools_nnp.utils.multiproc."""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from tstools_nnp.utils.multiproc import batch_dicts  # noqa: E402


class TestBatchDicts:
    def test_even_split(self):
        items = [{"v": i} for i in range(4)]
        batches = batch_dicts(items, 2)
        assert sum(len(b) for b in batches) == 4
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2

    def test_single_worker(self):
        items = [{"v": i} for i in range(5)]
        batches = batch_dicts(items, 1)
        assert sum(len(b) for b in batches) == 5

    def test_all_items_preserved(self):
        items = [{"v": i} for i in range(7)]
        batches = batch_dicts(items, 3)
        total = sum(len(b) for b in batches)
        assert total == 7

    def test_more_workers_than_items(self):
        items = [{"v": i} for i in range(2)]
        batches = batch_dicts(items, 5)
        total = sum(len(b) for b in batches)
        assert total == 2

    def test_batch_numbers_assigned(self):
        items = [{"v": i} for i in range(4)]
        batch_dicts(items, 2)
        # batch_dicts mutates the dicts in-place to add a 'batch' key
        assert all("batch" in item for item in items)

    def test_single_item(self):
        items = [{"v": 0}]
        batches = batch_dicts(items, 1)
        assert sum(len(b) for b in batches) == 1
