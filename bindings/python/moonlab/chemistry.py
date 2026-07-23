"""Moonlab quantum-chemistry bindings.

Thin ctypes layer over the C electronic-structure engine that turns a
molecular geometry into a qubit Hamiltonian.  Everything here is backed
by real C code -- there are no interpolated placeholder coefficients:

- :func:`h2_sto3g_pauli_coeffs` calls ``h2_sto3g_pauli_coeffs`` in
  ``src/algorithms/h2_sto3g.c``, which evaluates genuine STO-3G Gaussian
  integrals (own Boys/erf), reduces them through Slater-Condon to the
  seniority-zero space, and Jordan-Wigner maps the result to the five
  two-qubit Pauli coefficients ``[II, IZ, ZI, ZZ, XX]``.  The map is
  C-infinity smooth in the bond length, so the coefficients vary
  smoothly and a geometry derivative of the VQE energy is a real force.
- :class:`Hamiltonian` wraps ``pauli_hamiltonian_t*`` built by
  ``vqe_create_h2_hamiltonian`` / ``vqe_create_lih_hamiltonian`` and
  exposes the actual Pauli-string term list plus the exact ground state
  energy from ``vqe_exact_ground_state_energy``.
- :class:`Molecule` accepts an atom list and Cartesian coordinates (the
  documented ``Molecule(atoms=['H', 'H'], coordinates=...)`` surface) and
  dispatches to the correct pre-built molecular Hamiltonian.

Quick example -- the documented H2 workflow::

    >>> from moonlab.chemistry import Molecule, Hamiltonian
    >>> h2 = Molecule(atoms=['H', 'H'],
    ...               coordinates=[[0, 0, 0], [0, 0, 0.74]])
    >>> H = h2.get_hamiltonian(basis='sto-3g')
    >>> H.num_qubits
    2
    >>> round(h2.fci_energy(), 3)     # exact STO-3G ground state (Hartree)
    -1.137
    >>> H2 = Hamiltonian.h2_sto3g(bond_distance=0.74)
    >>> [(round(c, 4), p) for c, p in H2.terms]
    [(-1.0523, 'II'), (0.3979, 'IZ'), (-0.3979, 'ZI'), (-0.0113, 'ZZ'), (0.1809, 'XX')]

The VQE symbols are bound through isolated ``CFUNCTYPE`` prototypes
rather than by mutating ``_lib.<fn>.argtypes`` so this module never
collides with :mod:`moonlab.algorithms`, which treats the same
``pauli_hamiltonian_t*`` as an opaque handle.
"""

from __future__ import annotations

import ctypes
import math
from typing import List, Sequence, Tuple

import numpy as np

from .core import _lib

__all__ = [
    "Molecule",
    "Hamiltonian",
    "h2_sto3g_pauli_coeffs",
    "hartree_to_kcalmol",
    "H2_PAULI_LABELS",
    "CHEMICAL_ACCURACY_HARTREE",
]

# The five Pauli strings of the two-qubit STO-3G H2 Hamiltonian, in the
# order the C engine writes its coefficient array g[0..4].
H2_PAULI_LABELS: Tuple[str, ...] = ("II", "IZ", "ZI", "ZZ", "XX")

# 1 kcal/mol expressed in Hartree -- the standard "chemical accuracy"
# threshold used to judge a VQE result against the FCI reference.
CHEMICAL_ACCURACY_HARTREE = 0.0015936


# ---------------------------------------------------------------- #
# C struct layouts (mirror src/algorithms/vqe.h exactly).          #
# ---------------------------------------------------------------- #


class _CPauliTerm(ctypes.Structure):
    """Mirror of ``pauli_term_t`` (src/algorithms/vqe.h)."""

    _fields_ = [
        ("coefficient", ctypes.c_double),
        ("pauli_string", ctypes.c_char_p),
        ("num_qubits", ctypes.c_size_t),
    ]


class _CPauliHamiltonian(ctypes.Structure):
    """Mirror of ``pauli_hamiltonian_t`` (src/algorithms/vqe.h)."""

    _fields_ = [
        ("num_qubits", ctypes.c_size_t),
        ("num_terms", ctypes.c_size_t),
        ("terms", ctypes.POINTER(_CPauliTerm)),
        ("nuclear_repulsion", ctypes.c_double),
        ("molecule_name", ctypes.c_char_p),
        ("bond_distance", ctypes.c_double),
        ("hf_reference", ctypes.c_uint64),
    ]


_HamPtr = ctypes.POINTER(_CPauliHamiltonian)


# ---------------------------------------------------------------- #
# Isolated foreign-function bindings.                              #
#                                                                  #
# Binding through CFUNCTYPE prototypes (rather than assigning to    #
# _lib.<fn>.argtypes) gives each function its own restype/argtypes  #
# that no other module can clobber -- moonlab.algorithms binds the  #
# same pauli_hamiltonian symbols as opaque c_void_p handles.        #
# ---------------------------------------------------------------- #

_c_h2_coeffs = ctypes.CFUNCTYPE(
    None, ctypes.c_double, ctypes.POINTER(ctypes.c_double)
)(("h2_sto3g_pauli_coeffs", _lib))

_c_create_h2 = ctypes.CFUNCTYPE(_HamPtr, ctypes.c_double)(
    ("vqe_create_h2_hamiltonian", _lib)
)

_c_create_lih = ctypes.CFUNCTYPE(_HamPtr, ctypes.c_double)(
    ("vqe_create_lih_hamiltonian", _lib)
)

_c_ham_free = ctypes.CFUNCTYPE(None, _HamPtr)(("pauli_hamiltonian_free", _lib))

_c_exact_energy = ctypes.CFUNCTYPE(ctypes.c_double, _HamPtr)(
    ("vqe_exact_ground_state_energy", _lib)
)

_c_hartree_to_kcalmol = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)(
    ("vqe_hartree_to_kcalmol", _lib)
)


# ---------------------------------------------------------------- #
# Free functions.                                                  #
# ---------------------------------------------------------------- #


def h2_sto3g_pauli_coeffs(bond_distance: float) -> np.ndarray:
    """First-principles STO-3G H2 Pauli coefficients at a bond length.

    Calls the C routine ``h2_sto3g_pauli_coeffs`` and returns the five
    electronic coefficients ``[g_II, g_IZ, g_ZI, g_ZZ, g_XX]`` (Hartree)
    as a length-5 float64 array, in one-to-one correspondence with
    :data:`H2_PAULI_LABELS`.  The nuclear-repulsion energy is separate
    (see :attr:`Hamiltonian.nuclear_repulsion`).

    The coefficients are smooth (C-infinity) in ``bond_distance``, so a
    finite-difference derivative of them -- or of any energy built from
    them -- yields the interatomic force with no spurious kink.

    Parameters
    ----------
    bond_distance : float
        H-H internuclear distance in Angstroms.

    Returns
    -------
    numpy.ndarray, shape (5,)
        ``[II, IZ, ZI, ZZ, XX]`` coefficients in Hartree.
    """
    buf = (ctypes.c_double * 5)()
    _c_h2_coeffs(ctypes.c_double(float(bond_distance)), buf)
    return np.frombuffer(buf, dtype=np.float64).copy()


def hartree_to_kcalmol(energy_hartree: float) -> float:
    """Convert an energy in Hartree to kcal/mol via the C converter."""
    return float(_c_hartree_to_kcalmol(ctypes.c_double(float(energy_hartree))))


# ---------------------------------------------------------------- #
# Hamiltonian handle wrapper.                                      #
# ---------------------------------------------------------------- #


class Hamiltonian:
    """Qubit (Pauli sum-of-strings) molecular Hamiltonian.

    Wraps a ``pauli_hamiltonian_t*`` produced by the C engine:

        H = sum_i c_i P_i   +   nuclear_repulsion * I,

    where each ``P_i`` is a tensor product of single-qubit Pauli
    operators encoded as a string over ``{'I', 'X', 'Y', 'Z'}``.  The
    handle is freed automatically on garbage collection.

    Construct via the classmethods (:meth:`h2_sto3g`, :meth:`lih_sto3g`)
    or via :meth:`Molecule.get_hamiltonian`.
    """

    def __init__(self, handle):
        if not handle:
            raise MemoryError(
                "Hamiltonian constructor received a NULL pauli_hamiltonian_t* "
                "(the C factory returned NULL)"
            )
        self._h = handle

    def __del__(self):
        h = getattr(self, "_h", None)
        if h:
            _c_ham_free(h)
            self._h = None

    # -- constructors ------------------------------------------------ #

    @classmethod
    def h2_sto3g(cls, bond_distance: float = 0.74) -> "Hamiltonian":
        """Two-qubit STO-3G H2 Hamiltonian at ``bond_distance`` (Angstroms).

        The five terms ``[II, IZ, ZI, ZZ, XX]`` are anchored to the exact
        O'Malley et al. (PRX 6, 031007) values at equilibrium and carry
        the smooth first-principles STO-3G slope away from it.
        """
        return cls(_c_create_h2(ctypes.c_double(float(bond_distance))))

    @classmethod
    def lih_sto3g(cls, bond_distance: float = 1.5949) -> "Hamiltonian":
        """Four-qubit LiH Hamiltonian at ``bond_distance`` (Angstroms).

        Frozen-core (2 core electrons) active space, Jordan-Wigner mapped
        to 4 qubits.
        """
        return cls(_c_create_lih(ctypes.c_double(float(bond_distance))))

    # -- scalar properties ------------------------------------------- #

    @property
    def num_qubits(self) -> int:
        return int(self._h.contents.num_qubits)

    @property
    def num_terms(self) -> int:
        """Number of populated Pauli terms (matches ``len(self.terms)``).

        The C factories over-allocate the term array and fill only the
        physically meaningful slots, leaving the rest with a NULL
        ``pauli_string`` -- exactly the slots the C evaluator skips.  This
        counts the populated terms, not the raw allocation (see
        :attr:`allocated_terms`)."""
        return len(self.terms)

    @property
    def allocated_terms(self) -> int:
        """Raw ``pauli_hamiltonian_t.num_terms`` (the allocated slot count)."""
        return int(self._h.contents.num_terms)

    @property
    def nuclear_repulsion(self) -> float:
        """Nuclear-nuclear repulsion energy (Hartree), added to <H>."""
        return float(self._h.contents.nuclear_repulsion)

    @property
    def bond_distance(self) -> float:
        return float(self._h.contents.bond_distance)

    @property
    def molecule_name(self) -> str:
        name = self._h.contents.molecule_name
        return name.decode("utf-8", errors="replace") if name else ""

    @property
    def hf_reference(self) -> int:
        """Hartree-Fock reference bitstring (bit q set => qubit q is |1>)."""
        return int(self._h.contents.hf_reference)

    # -- term access ------------------------------------------------- #

    @property
    def terms(self) -> List[Tuple[float, str]]:
        """The Pauli terms as a list of ``(coefficient, pauli_string)``.

        Ordering matches the C engine.  For H2 this is exactly
        ``[(c_II, 'II'), (c_IZ, 'IZ'), (c_ZI, 'ZI'), (c_ZZ, 'ZZ'),
        (c_XX, 'XX')]``.
        """
        ham = self._h.contents
        out: List[Tuple[float, str]] = []
        for i in range(int(ham.num_terms)):
            term = ham.terms[i]
            pstr = term.pauli_string
            # Over-allocated but unpopulated slots carry a NULL pauli_string;
            # the C evaluator skips them, so we do too.
            if not pstr:
                continue
            out.append((float(term.coefficient), pstr.decode("ascii")))
        return out

    @property
    def coefficients(self) -> np.ndarray:
        """The term coefficients as a float64 array (Hartree)."""
        return np.array([c for c, _ in self.terms], dtype=np.float64)

    def exact_ground_state(self) -> float:
        """Exact ground-state energy including nuclear repulsion (Hartree).

        Direct diagonalization of the full Pauli Hamiltonian via
        ``vqe_exact_ground_state_energy`` -- an FCI-equivalent reference
        for the given basis and geometry.
        """
        return float(_c_exact_energy(self._h))

    def exact_ground_state_kcalmol(self) -> float:
        """:meth:`exact_ground_state` expressed in kcal/mol."""
        return hartree_to_kcalmol(self.exact_ground_state())

    def __repr__(self) -> str:
        return (
            f"Hamiltonian(molecule={self.molecule_name!r}, "
            f"num_qubits={self.num_qubits}, num_terms={self.num_terms}, "
            f"bond_distance={self.bond_distance:.4f})"
        )


# ---------------------------------------------------------------- #
# Molecule.                                                        #
# ---------------------------------------------------------------- #


# Element symbol -> canonical case, so 'h', 'H', 'li', 'Li' all resolve.
def _canonical_symbol(sym: str) -> str:
    s = str(sym).strip()
    if not s:
        raise ValueError("empty atom symbol")
    return s[0].upper() + s[1:].lower()


# Supported diatomic species keyed by the sorted tuple of element symbols.
_SUPPORTED_SPECIES = {
    ("H", "H"): "H2",
    ("H", "Li"): "LiH",
}

_SUPPORTED_BASES = {"sto-3g", "sto3g"}


class Molecule:
    """A molecule defined by its atoms and Cartesian coordinates.

    This is the documented entry point::

        h2 = Molecule(atoms=['H', 'H'],
                      coordinates=[[0, 0, 0], [0, 0, 0.74]])
        H = h2.get_hamiltonian(basis='sto-3g')

    Coordinates are in Angstroms.  Currently the pre-built C molecular
    Hamiltonians cover the diatomics H2 and LiH; the molecule's bond
    length (the internuclear distance) selects the point on the potential
    energy surface.

    Parameters
    ----------
    atoms : sequence of str
        Element symbols, e.g. ``['H', 'H']`` or ``['Li', 'H']``.
    coordinates : sequence of 3-vectors
        One ``(x, y, z)`` per atom, in Angstroms.
    charge : int, optional
        Total molecular charge (informational; must be 0 for the built-in
        neutral Hamiltonians).
    multiplicity : int, optional
        Spin multiplicity ``2S + 1`` (informational; the built-in
        Hamiltonians are singlets).
    """

    def __init__(
        self,
        atoms: Sequence[str],
        coordinates: Sequence[Sequence[float]],
        charge: int = 0,
        multiplicity: int = 1,
    ):
        atoms = list(atoms)
        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                "coordinates must be an (n_atoms, 3) array of xyz positions"
            )
        if len(atoms) != coords.shape[0]:
            raise ValueError(
                f"len(atoms)={len(atoms)} != len(coordinates)={coords.shape[0]}"
            )
        if len(atoms) < 2:
            raise ValueError("a molecule needs at least two atoms")

        self.atoms = [_canonical_symbol(a) for a in atoms]
        self.coordinates = coords
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)

        key = tuple(sorted(self.atoms))
        species = _SUPPORTED_SPECIES.get(key)
        if species is None:
            supported = ", ".join(
                "".join(k) for k in _SUPPORTED_SPECIES
            )
            raise ValueError(
                f"unsupported molecule {self.atoms!r}; the built-in "
                f"Hamiltonians cover: {supported}"
            )
        self.species = species

    # -- convenience constructors ------------------------------------ #

    @classmethod
    def h2(cls, bond_distance: float = 0.74) -> "Molecule":
        """H2 aligned on the z-axis at the given bond length (Angstroms)."""
        return cls(
            atoms=["H", "H"],
            coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, float(bond_distance)]],
        )

    @classmethod
    def lih(cls, bond_distance: float = 1.5949) -> "Molecule":
        """LiH aligned on the z-axis at the given bond length (Angstroms)."""
        return cls(
            atoms=["Li", "H"],
            coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, float(bond_distance)]],
        )

    # -- geometry ---------------------------------------------------- #

    @property
    def bond_distance(self) -> float:
        """Distance between the two nearest atoms (Angstroms).

        For a diatomic this is the bond length used to select the point on
        the potential energy surface.
        """
        if len(self.atoms) == 2:
            return float(np.linalg.norm(self.coordinates[0] - self.coordinates[1]))
        # General case: shortest interatomic separation.
        best = math.inf
        for i in range(len(self.atoms)):
            for j in range(i + 1, len(self.atoms)):
                d = float(np.linalg.norm(self.coordinates[i] - self.coordinates[j]))
                best = min(best, d)
        return best

    # -- Hamiltonian dispatch ---------------------------------------- #

    def get_hamiltonian(self, basis: str = "sto-3g") -> Hamiltonian:
        """Build the qubit Hamiltonian for this geometry.

        Parameters
        ----------
        basis : str
            Basis set. Only the minimal ``'sto-3g'`` basis is wired to the
            C engine today.

        Returns
        -------
        Hamiltonian
            Jordan-Wigner encoded Pauli Hamiltonian at this molecule's
            bond length.
        """
        if str(basis).lower().replace("_", "-") not in _SUPPORTED_BASES:
            raise ValueError(
                f"unsupported basis {basis!r}; supported: 'sto-3g'"
            )
        r = self.bond_distance
        if self.species == "H2":
            return Hamiltonian.h2_sto3g(bond_distance=r)
        if self.species == "LiH":
            return Hamiltonian.lih_sto3g(bond_distance=r)
        raise ValueError(f"no Hamiltonian factory for species {self.species!r}")

    def fci_energy(self, basis: str = "sto-3g") -> float:
        """Exact (FCI-equivalent) ground-state energy for this geometry.

        Builds the Hamiltonian and diagonalizes it via
        :meth:`Hamiltonian.exact_ground_state`.  Returns Hartree.
        """
        return self.get_hamiltonian(basis=basis).exact_ground_state()

    def __repr__(self) -> str:
        return (
            f"Molecule(species={self.species!r}, atoms={self.atoms!r}, "
            f"bond_distance={self.bond_distance:.4f})"
        )
