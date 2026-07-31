"""Sinter decoder plugins backed by moonlab's union-find decoders.

Registers two decoders with `sinter <https://pypi.org/project/sinter/>`_:

===========================  ====================================
name                         decoder
===========================  ====================================
``moonlab_uf``               Delfosse-Nickerson union-find
``moonlab_uf_correlated``    two-pass correlated union-find, which
                             consumes the ``^`` decompositions
                             ``decompose_errors=True`` emits
===========================  ====================================

From Python::

    import sinter
    from moonlab.qec.sinter import make_custom_decoders

    stats = sinter.collect(
        num_workers=8,
        tasks=tasks,
        decoders=['moonlab_uf_correlated', 'pymatching'],
        custom_decoders=make_custom_decoders(),
        max_shots=1_000_000,
    )

From the command line::

    sinter collect --decoders moonlab_uf \\
        --custom_decoders_module_function moonlab.qec.sinter:make_custom_decoders \\
        --circuits *.stim --max_shots 100000 --save_resume_filepath out.csv

``sinter`` is a development / test dependency, never a runtime one:
``import moonlab`` and ``import moonlab.qec`` need numpy alone, and this
module is the only place in the package that touches sinter.  It is not
imported by ``moonlab.qec.__init__``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

try:
    import sinter
except ImportError as _exc:  # pragma: no cover - exercised only without sinter
    raise ImportError(
        "moonlab.qec.sinter requires the 'sinter' package "
        "(dev extra: pip install moonlab[qec])"
    ) from _exc

from .dem import DetectorErrorModel

__all__ = [
    "MoonlabCompiledDecoder",
    "MoonlabUFDecoder",
    "MoonlabCorrelatedDecoder",
    "make_custom_decoders",
]


class MoonlabCompiledDecoder(sinter.CompiledDecoder):
    """A moonlab decoder compiled for one detector error model.

    Lives entirely inside the worker process: sinter builds it by
    calling :meth:`MoonlabUFDecoder.compile_decoder_for_dem` after the
    task has been shipped, so no ctypes handle ever has to be pickled.
    """

    def __init__(self, decoder) -> None:
        self._decoder = decoder

    def decode_shots_bit_packed(self, *,
                                bit_packed_detection_event_data: np.ndarray
                                ) -> np.ndarray:
        return self._decoder.decode_shots_bit_packed(
            bit_packed_detection_event_data, threads=1)


class _MoonlabDecoderBase(sinter.Decoder):
    """Shared plumbing; picklable because it holds no native state.

    sinter ships decoder objects to worker processes, so the instance
    must survive a pickle round trip.  Everything native is built inside
    :meth:`compile_decoder_for_dem`, which runs in the worker.
    """

    #: two-pass correlated decoding.
    correlated = False

    def compile_decoder_for_dem(self, *, dem) -> sinter.CompiledDecoder:
        model = DetectorErrorModel.from_text(str(dem))
        return MoonlabCompiledDecoder(
            model.make_decoder(correlated=self.correlated))

    def __eq__(self, other) -> bool:
        return type(other) is type(self)

    def __hash__(self) -> int:
        return hash(type(self))

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class MoonlabUFDecoder(_MoonlabDecoderBase):
    """Plain Delfosse-Nickerson union-find decoding."""

    correlated = False


class MoonlabCorrelatedDecoder(_MoonlabDecoderBase):
    """Two-pass correlated union-find decoding.

    Stim's ``decompose_errors=True`` emits mechanisms like
    ``error(p) D1 D2 ^ D3 D4``: one physical fault whose graphlike
    components always fire together.  A matching decoder throws that
    correlation away; this one decodes twice, re-weighting a mechanism's
    partner components by their conditional probabilities once pass 1
    has used one of them.
    """

    correlated = True


def make_custom_decoders() -> Dict[str, sinter.Decoder]:
    """The ``custom_decoders`` mapping to hand to :func:`sinter.collect`."""
    return {
        "moonlab_uf": MoonlabUFDecoder(),
        "moonlab_uf_correlated": MoonlabCorrelatedDecoder(),
    }
