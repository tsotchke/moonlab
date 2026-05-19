"""Python tests for moonlab.decoder (v0.7.3 binding)."""

from __future__ import annotations

import pytest

from moonlab.decoder import (
    DecoderSlot, DecoderNotBuiltError, decode,
    slot_available, slot_name,
)


def test_slot_naming():
    assert slot_name(DecoderSlot.GREEDY) == "greedy"
    assert slot_name(DecoderSlot.MWPM_EXACT) == "mwpm_exact"
    assert slot_name(DecoderSlot.SBNN) == "sbnn"
    assert slot_name(DecoderSlot.LIBIRREP_SS) == "libirrep_single_shot"
    assert slot_name(DecoderSlot.PYMATCHING) == "pymatching"


def test_slot_availability():
    """GREEDY + MWPM_EXACT always available; SBNN + PYMATCHING never;
    LIBIRREP_SS depends on build flags."""
    assert slot_available(DecoderSlot.GREEDY) is True
    assert slot_available(DecoderSlot.MWPM_EXACT) is True
    assert slot_available(DecoderSlot.SBNN) is False
    assert slot_available(DecoderSlot.PYMATCHING) is False
    # LIBIRREP_SS: either OK is acceptable.


def test_greedy_zero_syndrome():
    d = 3
    corr = decode(DecoderSlot.GREEDY,
                  distance=d, num_qubits=2 * d * d, is_toric=True,
                  syndromes=[0] * (d * d))
    assert sum(corr) == 0


def test_greedy_two_adjacent_defects():
    d = 3
    s = [0] * (d * d)
    s[0] = 1; s[1] = 1
    corr = decode(DecoderSlot.GREEDY,
                  distance=d, num_qubits=2 * d * d, is_toric=True,
                  syndromes=s)
    assert sum(corr) == 1


def test_mwpm_exact_l1_4():
    d = 5
    s = [0] * (d * d)
    s[0]  = 1  # (0, 0)
    s[12] = 1  # (2, 2)
    corr = decode(DecoderSlot.MWPM_EXACT,
                  distance=d, num_qubits=2 * d * d, is_toric=True,
                  syndromes=s)
    # Single geodesic, L1 distance 4 -> exactly 4 flips.
    assert sum(corr) == 4


def test_sbnn_not_built():
    d = 3
    with pytest.raises(DecoderNotBuiltError):
        decode(DecoderSlot.SBNN,
               distance=d, num_qubits=2 * d * d, is_toric=True,
               syndromes=[0] * (d * d))
