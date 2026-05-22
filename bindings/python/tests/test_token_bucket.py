"""Tests for moonlab.token_bucket.TokenBucket.

Mirrors the C-side ``tests/unit/test_token_bucket.c`` cases so the
two implementations stay in sync.
"""

import threading
import time

import pytest

from moonlab.token_bucket import TokenBucket


def test_init_starts_full():
    b = TokenBucket(burst=100, refill_per_sec=10)
    assert b.peek() == 100


def test_take_drains():
    b = TokenBucket(burst=10, refill_per_sec=0)
    for _ in range(10):
        assert b.take(1)
    assert not b.take(1)
    assert b.peek() == 0


def test_refill_clips_at_burst():
    b = TokenBucket(burst=5, refill_per_sec=0)
    assert b.take(5)
    assert b.peek() == 0
    b.refill(1000)
    assert b.peek() == 5


def test_time_refill_accrues():
    b = TokenBucket(burst=100, refill_per_sec=100)  # 100 tok/sec
    assert b.take(100)
    assert b.peek() == 0
    time.sleep(0.25)
    p = b.peek()
    # CI runners can overrun the 250ms sleep -- 250ms wall can become
    # 400-500ms under load, giving 40-50 tokens.  Check the floor and
    # the burst cap (invariants) rather than a tight window.  Same
    # hardening as the C-side unit_token_bucket and Rust-side
    # time_refill_accrues tests.
    assert p >= 20, f"after >=250ms at 100/s, peek = {p} (expected >= 20)"
    assert p <= 100, f"burst cap violated, peek = {p} (expected <= 100)"
    assert b.take(10)


def test_thread_safety_exact_balance():
    """8 threads * 1000 attempts on burst=500 with no refill: exactly
    500 takes should succeed across 8000 attempts.  Same invariant as
    the C primitive's racing-take test."""
    b = TokenBucket(burst=500, refill_per_sec=0)
    counts = [0] * 8

    def worker(i):
        local = 0
        for _ in range(1000):
            if b.take(1):
                local += 1
        counts[i] = local

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()

    total = sum(counts)
    assert total == 500, f"expected 500 takes across 8 threads, got {total}"
    assert b.peek() == 0


def test_zero_cost_take_is_free():
    b = TokenBucket(burst=1, refill_per_sec=0)
    assert b.take(1)
    assert b.peek() == 0
    assert b.take(0)
    assert b.peek() == 0


def test_invalid_init():
    with pytest.raises(ValueError):
        TokenBucket(burst=0, refill_per_sec=1)
    with pytest.raises(ValueError):
        TokenBucket(burst=-1, refill_per_sec=1)
    with pytest.raises(ValueError):
        TokenBucket(burst=10, refill_per_sec=-1)
