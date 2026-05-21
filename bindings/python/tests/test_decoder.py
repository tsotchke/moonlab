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
    """GREEDY + MWPM_EXACT always available; SBNN gated by build flag;
    PYMATCHING + LIBIRREP_SS depend on optional linkage."""
    assert slot_available(DecoderSlot.GREEDY) is True
    assert slot_available(DecoderSlot.MWPM_EXACT) is True
    # SBNN requires QSIM_ENABLE_SBNN=ON; default OFF.
    assert slot_available(DecoderSlot.SBNN) is False
    # PYMATCHING + LIBIRREP_SS: either availability is acceptable.
    _ = slot_available(DecoderSlot.PYMATCHING)
    _ = slot_available(DecoderSlot.LIBIRREP_SS)


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


# ---- Runtime decoder registry tests (v1.0.3) ----

from moonlab.decoder import (
    decode_by_name, list_decoders, lookup_decoder,
    register_decoder, unregister_decoder,
)


def test_list_decoders_contains_builtins():
    names = list_decoders()
    for required in ("greedy", "mwpm_exact", "sbnn",
                     "libirrep_single_shot", "pymatching"):
        assert required in names, f"built-in {required!r} missing from registry"


def test_lookup_decoder_returns_known_entries():
    g = lookup_decoder("greedy")
    assert g is not None
    assert g["name"] == "greedy"
    assert lookup_decoder("does-not-exist") is None


def test_decode_by_name_matches_decode():
    d = 3
    s = [0] * (d * d)
    s[0] = 1; s[1] = 1
    via_enum = decode(DecoderSlot.GREEDY,
                      distance=d, num_qubits=2 * d * d, is_toric=True,
                      syndromes=s)
    via_name = decode_by_name("greedy",
                              distance=d, num_qubits=2 * d * d, is_toric=True,
                              syndromes=s)
    assert via_enum == via_name


def test_register_custom_decoder_round_trips():
    """Register a Python decoder, dispatch by name, verify it ran."""
    sentinel = 0x42

    def sentinel_decoder(distance, num_qubits, is_toric, syndromes):
        # Write the sentinel into the first byte; zero everywhere else.
        out = [0] * num_qubits
        out[0] = sentinel
        return out

    register_decoder("py-sentinel-test", sentinel_decoder,
                     description="round-trip sanity decoder")
    try:
        assert "py-sentinel-test" in list_decoders()
        entry = lookup_decoder("py-sentinel-test")
        assert entry is not None
        assert "round-trip" in entry["description"]

        d = 3
        corr = decode_by_name("py-sentinel-test",
                              distance=d, num_qubits=2 * d * d, is_toric=True,
                              syndromes=[0] * (d * d))
        assert corr[0] == sentinel
        assert all(c == 0 for c in corr[1:])
    finally:
        unregister_decoder("py-sentinel-test")
    assert "py-sentinel-test" not in list_decoders()


def test_unregister_unknown_raises():
    """Removing a name that was never registered must error so callers
    notice typos instead of silently failing."""
    from moonlab.decoder import DecoderError
    with pytest.raises(DecoderError):
        unregister_decoder("definitely-not-a-real-decoder")
