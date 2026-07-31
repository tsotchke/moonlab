"""Quantum error correction interop with the Stim ecosystem.

This subpackage reads and writes the two formats the QEC community
publishes in -- Stim's `.stim` circuits and `.dem` detector error
models -- so moonlab drops into an existing Stim / Sinter / PyMatching
harness without a translation layer, and so moonlab's sampler and
decoders can be differentially tested against that ecosystem on its own
circuits.

Contents:
    :class:`StimCircuit`         parse / serialise / lower / sample `.stim`
    :class:`DetectorErrorModel`  parse / serialise `.dem` as an edge list
    :class:`UFDecoder`           moonlab's union-find decoder
    :class:`StimFormatError`     parse failure carrying code, line, message

numpy is the only runtime dependency.  ``moonlab.qec.sinter`` is NOT
imported here: it is the one module that needs the `sinter` development
package, and importing it eagerly would make this subpackage unusable on
a plain install.  Import it explicitly when plugging moonlab's decoders
into a sinter collection run.

Example:
    >>> import stim                                  # doctest: +SKIP
    >>> from moonlab.qec import StimCircuit, DetectorErrorModel
    >>> circuit = stim.Circuit.generated(            # doctest: +SKIP
    ...     'surface_code:rotated_memory_z', distance=3, rounds=3,
    ...     after_clifford_depolarization=0.005)
    >>> c = StimCircuit.from_text(str(circuit))      # doctest: +SKIP
    >>> dem = DetectorErrorModel.from_text(          # doctest: +SKIP
    ...     str(circuit.detector_error_model(decompose_errors=True)))
    >>> decoder = dem.make_decoder(correlated=True)  # doctest: +SKIP
    >>> det, obs = c.sample_detectors(10000, seed=1) # doctest: +SKIP
    >>> pred = decoder.decode_batch(det)             # doctest: +SKIP
"""

from ._ffi import PF_OP, StimFormatError, StimStatus
from .dem import UF_BOUNDARY, DetectorErrorModel, UFDecoder, text_from_edges
from .stim_io import StimCircuit

__all__ = [
    "StimCircuit",
    "DetectorErrorModel",
    "UFDecoder",
    "StimFormatError",
    "StimStatus",
    "PF_OP",
    "UF_BOUNDARY",
    "text_from_edges",
]
