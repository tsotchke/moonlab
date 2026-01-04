"""
Moonlab Algorithms - Python wrappers for quantum algorithms

VQE, QAOA, Grover, Bell Tests - Full C library bindings
"""

import ctypes
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from .core import _lib, QuantumState, QuantumError, CQuantumState

# ============================================================================
# C STRUCTURE DEFINITIONS
# ============================================================================

class CMolecularHamiltonian(ctypes.Structure):
    """Opaque structure for molecular Hamiltonian"""
    _fields_ = [
        ("terms", ctypes.c_void_p),
        ("num_terms", ctypes.c_size_t),
        ("num_qubits", ctypes.c_size_t),
    ]

class CVQEAnsatz(ctypes.Structure):
    """Opaque structure for VQE ansatz"""
    _fields_ = [
        ("num_qubits", ctypes.c_size_t),
        ("num_layers", ctypes.c_size_t),
        ("num_params", ctypes.c_size_t),
        ("ansatz_type", ctypes.c_int),
    ]

class CVQEOptimizer(ctypes.Structure):
    """Opaque structure for VQE optimizer"""
    _fields_ = [
        ("type", ctypes.c_int),
        ("learning_rate", ctypes.c_double),
        ("max_iter", ctypes.c_int),
    ]

class CVQESolver(ctypes.Structure):
    """Opaque structure for VQE solver"""
    pass

class CVQEResult(ctypes.Structure):
    """Result from VQE solver"""
    _fields_ = [
        ("energy", ctypes.c_double),
        ("optimal_params", ctypes.POINTER(ctypes.c_double)),
        ("num_params", ctypes.c_size_t),
        ("converged", ctypes.c_int),
        ("num_iterations", ctypes.c_int),
        ("final_gradient_norm", ctypes.c_double),
    ]

class CIsingModel(ctypes.Structure):
    """Opaque structure for Ising model"""
    _fields_ = [
        ("num_qubits", ctypes.c_size_t),
        ("J", ctypes.POINTER(ctypes.c_double)),  # Coupling matrix
        ("h", ctypes.POINTER(ctypes.c_double)),  # Field vector
    ]

class CGraph(ctypes.Structure):
    """Graph structure for QAOA"""
    _fields_ = [
        ("num_vertices", ctypes.c_size_t),
        ("num_edges", ctypes.c_size_t),
        ("edges", ctypes.c_void_p),
    ]

class CQAOASolver(ctypes.Structure):
    """Opaque structure for QAOA solver"""
    pass

class CQAOAResult(ctypes.Structure):
    """Result from QAOA solver"""
    _fields_ = [
        ("expectation", ctypes.c_double),
        ("optimal_gamma", ctypes.POINTER(ctypes.c_double)),
        ("optimal_beta", ctypes.POINTER(ctypes.c_double)),
        ("num_layers", ctypes.c_size_t),
        ("best_bitstring", ctypes.c_uint64),
        ("best_cost", ctypes.c_double),
        ("converged", ctypes.c_int),
    ]

class CGroverConfig(ctypes.Structure):
    """Configuration for Grover search"""
    _fields_ = [
        ("marked_state", ctypes.c_uint64),
        ("num_qubits", ctypes.c_size_t),
        ("num_iterations", ctypes.c_size_t),
        ("use_optimal_iterations", ctypes.c_int),
    ]

class CGroverResult(ctypes.Structure):
    """Result from Grover search"""
    _fields_ = [
        ("found_state", ctypes.c_uint64),
        ("success", ctypes.c_int),
        ("iterations_used", ctypes.c_size_t),
        ("probability", ctypes.c_double),
    ]

class CBellMeasurementSettings(ctypes.Structure):
    """Settings for Bell measurement"""
    _fields_ = [
        ("a", ctypes.c_double),
        ("a_prime", ctypes.c_double),
        ("b", ctypes.c_double),
        ("b_prime", ctypes.c_double),
    ]

class CBellTestResult(ctypes.Structure):
    """Result from Bell test"""
    _fields_ = [
        ("correlations", ctypes.c_double * 4),  # E(a,b), E(a,b'), E(a',b), E(a',b')
        ("chsh_parameter", ctypes.c_double),
        ("violates_classical", ctypes.c_int),
        ("num_measurements", ctypes.c_size_t),
        ("variance", ctypes.c_double),
    ]

# ============================================================================
# C FUNCTION SIGNATURES
# ============================================================================

# VQE functions
_lib.molecular_hamiltonian_create.argtypes = [ctypes.c_size_t]
_lib.molecular_hamiltonian_create.restype = ctypes.POINTER(CMolecularHamiltonian)

_lib.molecular_hamiltonian_free.argtypes = [ctypes.POINTER(CMolecularHamiltonian)]
_lib.molecular_hamiltonian_free.restype = None

_lib.molecular_hamiltonian_add_term.argtypes = [
    ctypes.POINTER(CMolecularHamiltonian),
    ctypes.c_double,  # coefficient
    ctypes.c_char_p,  # pauli_string (e.g., "XZIY")
]
_lib.molecular_hamiltonian_add_term.restype = ctypes.c_int

_lib.vqe_create_h2_hamiltonian.argtypes = [ctypes.c_double]  # bond_distance
_lib.vqe_create_h2_hamiltonian.restype = ctypes.POINTER(CMolecularHamiltonian)

_lib.vqe_create_lih_hamiltonian.argtypes = [ctypes.c_double]  # bond_distance
_lib.vqe_create_lih_hamiltonian.restype = ctypes.POINTER(CMolecularHamiltonian)

_lib.vqe_create_h2o_hamiltonian.argtypes = []
_lib.vqe_create_h2o_hamiltonian.restype = ctypes.POINTER(CMolecularHamiltonian)

_lib.vqe_create_hardware_efficient_ansatz.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
_lib.vqe_create_hardware_efficient_ansatz.restype = ctypes.POINTER(CVQEAnsatz)

_lib.vqe_ansatz_free.argtypes = [ctypes.POINTER(CVQEAnsatz)]
_lib.vqe_ansatz_free.restype = None

_lib.vqe_optimizer_create.argtypes = [ctypes.c_int]  # type
_lib.vqe_optimizer_create.restype = ctypes.POINTER(CVQEOptimizer)

_lib.vqe_optimizer_free.argtypes = [ctypes.POINTER(CVQEOptimizer)]
_lib.vqe_optimizer_free.restype = None

_lib.vqe_solver_create.argtypes = [
    ctypes.POINTER(CMolecularHamiltonian),
    ctypes.POINTER(CVQEAnsatz),
    ctypes.POINTER(CVQEOptimizer),
]
_lib.vqe_solver_create.restype = ctypes.POINTER(CVQESolver)

_lib.vqe_solver_free.argtypes = [ctypes.POINTER(CVQESolver)]
_lib.vqe_solver_free.restype = None

_lib.vqe_solve.argtypes = [ctypes.POINTER(CVQESolver)]
_lib.vqe_solve.restype = CVQEResult

_lib.vqe_compute_energy.argtypes = [
    ctypes.POINTER(CVQESolver),
    ctypes.POINTER(ctypes.c_double),  # params
    ctypes.c_size_t,  # num_params
]
_lib.vqe_compute_energy.restype = ctypes.c_double

_lib.vqe_hartree_to_kcalmol.argtypes = [ctypes.c_double]
_lib.vqe_hartree_to_kcalmol.restype = ctypes.c_double

# QAOA functions
_lib.ising_model_create.argtypes = [ctypes.c_size_t]
_lib.ising_model_create.restype = ctypes.POINTER(CIsingModel)

_lib.ising_model_free.argtypes = [ctypes.POINTER(CIsingModel)]
_lib.ising_model_free.restype = None

_lib.ising_model_set_coupling.argtypes = [
    ctypes.POINTER(CIsingModel),
    ctypes.c_size_t, ctypes.c_size_t,  # i, j
    ctypes.c_double,  # value
]
_lib.ising_model_set_coupling.restype = ctypes.c_int

_lib.ising_model_set_field.argtypes = [
    ctypes.POINTER(CIsingModel),
    ctypes.c_size_t,  # i
    ctypes.c_double,  # value
]
_lib.ising_model_set_field.restype = ctypes.c_int

_lib.ising_model_evaluate.argtypes = [ctypes.POINTER(CIsingModel), ctypes.c_uint64]
_lib.ising_model_evaluate.restype = ctypes.c_double

_lib.graph_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
_lib.graph_create.restype = ctypes.POINTER(CGraph)

_lib.graph_free.argtypes = [ctypes.POINTER(CGraph)]
_lib.graph_free.restype = None

_lib.graph_add_edge.argtypes = [
    ctypes.POINTER(CGraph),
    ctypes.c_size_t,  # edge_idx
    ctypes.c_int, ctypes.c_int,  # u, v
    ctypes.c_double,  # weight
]
_lib.graph_add_edge.restype = ctypes.c_int

_lib.ising_encode_maxcut.argtypes = [ctypes.POINTER(CGraph)]
_lib.ising_encode_maxcut.restype = ctypes.POINTER(CIsingModel)

_lib.qaoa_solver_create.argtypes = [
    ctypes.POINTER(CIsingModel),
    ctypes.c_size_t,  # num_layers
]
_lib.qaoa_solver_create.restype = ctypes.POINTER(CQAOASolver)

_lib.qaoa_solver_free.argtypes = [ctypes.POINTER(CQAOASolver)]
_lib.qaoa_solver_free.restype = None

_lib.qaoa_solve.argtypes = [ctypes.POINTER(CQAOASolver)]
_lib.qaoa_solve.restype = CQAOAResult

_lib.qaoa_compute_expectation.argtypes = [
    ctypes.POINTER(CQAOASolver),
    ctypes.POINTER(ctypes.c_double),  # gamma
    ctypes.POINTER(ctypes.c_double),  # beta
    ctypes.c_size_t,  # num_layers
]
_lib.qaoa_compute_expectation.restype = ctypes.c_double

# Grover functions
_lib.grover_search.argtypes = [
    ctypes.POINTER(CQuantumState),
    ctypes.POINTER(CGroverConfig),
    ctypes.c_void_p,  # entropy context
]
_lib.grover_search.restype = CGroverResult

_lib.grover_optimal_iterations.argtypes = [ctypes.c_size_t]
_lib.grover_optimal_iterations.restype = ctypes.c_size_t

_lib.grover_oracle.argtypes = [ctypes.POINTER(CQuantumState), ctypes.c_uint64]
_lib.grover_oracle.restype = ctypes.c_int

_lib.grover_diffusion.argtypes = [ctypes.POINTER(CQuantumState)]
_lib.grover_diffusion.restype = ctypes.c_int

_lib.grover_iteration.argtypes = [ctypes.POINTER(CQuantumState), ctypes.c_uint64]
_lib.grover_iteration.restype = ctypes.c_int

# Bell test functions
_lib.create_bell_state.argtypes = [
    ctypes.POINTER(CQuantumState),
    ctypes.c_int, ctypes.c_int,  # qubit1, qubit2
    ctypes.c_int,  # type
]
_lib.create_bell_state.restype = ctypes.c_int

_lib.bell_test_chsh.argtypes = [
    ctypes.POINTER(CQuantumState),
    ctypes.c_int, ctypes.c_int,  # qubit_a, qubit_b
    ctypes.POINTER(CBellMeasurementSettings),
    ctypes.c_size_t,  # num_measurements
    ctypes.c_void_p,  # entropy context
]
_lib.bell_test_chsh.restype = CBellTestResult

_lib.bell_get_optimal_settings.argtypes = [ctypes.POINTER(CBellMeasurementSettings)]
_lib.bell_get_optimal_settings.restype = None

_lib.calculate_chsh_parameter.argtypes = [ctypes.c_double * 4]
_lib.calculate_chsh_parameter.restype = ctypes.c_double


# ============================================================================
# VQE CLASS
# ============================================================================

class VQE:
    """
    Variational Quantum Eigensolver for molecular ground state computation

    Supports H2, LiH, and H2O molecules, with hardware-efficient ansatz.
    Uses gradient-based optimization (Adam, L-BFGS, COBYLA).

    Example:
        vqe = VQE(num_qubits=4, num_layers=2)
        result = vqe.solve_h2(bond_distance=0.74)
        print(f"Ground state energy: {result['energy']} Hartree")
    """

    # Optimizer types (matching C enum)
    OPTIMIZER_ADAM = 0
    OPTIMIZER_GRADIENT_DESCENT = 1
    OPTIMIZER_LBFGS = 2
    OPTIMIZER_COBYLA = 3

    def __init__(self, num_qubits: int, num_layers: int = 2,
                 optimizer_type: int = OPTIMIZER_ADAM):
        """
        Initialize VQE solver

        Args:
            num_qubits: Number of qubits (determines molecule complexity)
            num_layers: Ansatz circuit depth
            optimizer_type: Optimization algorithm (ADAM, L-BFGS, COBYLA)
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.optimizer_type = optimizer_type

        # Create ansatz
        self._ansatz = _lib.vqe_create_hardware_efficient_ansatz(
            ctypes.c_size_t(num_qubits),
            ctypes.c_size_t(num_layers)
        )
        if not self._ansatz:
            raise QuantumError("Failed to create VQE ansatz")

        # Create optimizer
        self._optimizer = _lib.vqe_optimizer_create(ctypes.c_int(optimizer_type))
        if not self._optimizer:
            _lib.vqe_ansatz_free(self._ansatz)
            raise QuantumError("Failed to create VQE optimizer")

        self._hamiltonian = None
        self._solver = None

    def __del__(self):
        """Clean up C resources"""
        if hasattr(self, '_solver') and self._solver:
            _lib.vqe_solver_free(self._solver)
        if hasattr(self, '_optimizer') and self._optimizer:
            _lib.vqe_optimizer_free(self._optimizer)
        if hasattr(self, '_ansatz') and self._ansatz:
            _lib.vqe_ansatz_free(self._ansatz)
        if hasattr(self, '_hamiltonian') and self._hamiltonian:
            _lib.molecular_hamiltonian_free(self._hamiltonian)

    def solve_h2(self, bond_distance: float = 0.74) -> Dict[str, Any]:
        """
        Find ground state of H2 molecule

        Args:
            bond_distance: H-H bond distance in Angstroms (default: 0.74)

        Returns:
            Dict with 'energy' (Hartree), 'optimal_params', 'converged'
        """
        if self._hamiltonian:
            _lib.molecular_hamiltonian_free(self._hamiltonian)

        self._hamiltonian = _lib.vqe_create_h2_hamiltonian(
            ctypes.c_double(bond_distance)
        )
        if not self._hamiltonian:
            raise QuantumError("Failed to create H2 Hamiltonian")

        return self._solve()

    def solve_lih(self, bond_distance: float = 1.6) -> Dict[str, Any]:
        """
        Find ground state of LiH molecule

        Args:
            bond_distance: Li-H bond distance in Angstroms (default: 1.6)

        Returns:
            Dict with 'energy' (Hartree), 'optimal_params', 'converged'
        """
        if self._hamiltonian:
            _lib.molecular_hamiltonian_free(self._hamiltonian)

        self._hamiltonian = _lib.vqe_create_lih_hamiltonian(
            ctypes.c_double(bond_distance)
        )
        if not self._hamiltonian:
            raise QuantumError("Failed to create LiH Hamiltonian")

        return self._solve()

    def solve_h2o(self) -> Dict[str, Any]:
        """
        Find ground state of H2O molecule

        Returns:
            Dict with 'energy' (Hartree), 'optimal_params', 'converged'
        """
        if self._hamiltonian:
            _lib.molecular_hamiltonian_free(self._hamiltonian)

        self._hamiltonian = _lib.vqe_create_h2o_hamiltonian()
        if not self._hamiltonian:
            raise QuantumError("Failed to create H2O Hamiltonian")

        return self._solve()

    def solve(self, hamiltonian: 'MolecularHamiltonian') -> Dict[str, Any]:
        """
        Find ground state for custom Hamiltonian

        Args:
            hamiltonian: MolecularHamiltonian object

        Returns:
            Dict with 'energy' (Hartree), 'optimal_params', 'converged'
        """
        self._hamiltonian = hamiltonian._ptr
        return self._solve()

    def _solve(self) -> Dict[str, Any]:
        """Internal solve method"""
        if self._solver:
            _lib.vqe_solver_free(self._solver)

        self._solver = _lib.vqe_solver_create(
            self._hamiltonian,
            self._ansatz,
            self._optimizer
        )
        if not self._solver:
            raise QuantumError("Failed to create VQE solver")

        result = _lib.vqe_solve(self._solver)

        # Extract optimal parameters
        params = []
        if result.optimal_params and result.num_params > 0:
            for i in range(result.num_params):
                params.append(result.optimal_params[i])

        return {
            'energy': result.energy,
            'energy_kcal_mol': _lib.vqe_hartree_to_kcalmol(result.energy),
            'optimal_params': np.array(params),
            'converged': bool(result.converged),
            'num_iterations': result.num_iterations,
            'gradient_norm': result.final_gradient_norm,
        }

    def compute_energy(self, params: np.ndarray) -> float:
        """
        Compute energy for given parameters

        Args:
            params: Parameter array

        Returns:
            Energy in Hartree
        """
        if not self._solver:
            raise QuantumError("Must call solve() first to set up the problem")

        c_params = (ctypes.c_double * len(params))(*params)
        return _lib.vqe_compute_energy(
            self._solver,
            c_params,
            ctypes.c_size_t(len(params))
        )


class MolecularHamiltonian:
    """
    Custom molecular Hamiltonian for VQE

    Build Hamiltonians as sums of Pauli strings with coefficients.

    Example:
        H = MolecularHamiltonian(4)
        H.add_term(0.5, "ZIZI")   # ZZ interaction on qubits 0,2
        H.add_term(-0.3, "XIXI")  # XX interaction
    """

    def __init__(self, num_qubits: int):
        """Create empty Hamiltonian for num_qubits qubits"""
        self._ptr = _lib.molecular_hamiltonian_create(ctypes.c_size_t(num_qubits))
        if not self._ptr:
            raise QuantumError("Failed to create Hamiltonian")
        self.num_qubits = num_qubits

    def __del__(self):
        if hasattr(self, '_ptr') and self._ptr:
            _lib.molecular_hamiltonian_free(self._ptr)

    def add_term(self, coefficient: float, pauli_string: str) -> None:
        """
        Add a Pauli term to the Hamiltonian

        Args:
            coefficient: Real coefficient
            pauli_string: String of I, X, Y, Z (one per qubit)
        """
        if len(pauli_string) != self.num_qubits:
            raise ValueError(f"Pauli string must have {self.num_qubits} characters")

        ret = _lib.molecular_hamiltonian_add_term(
            self._ptr,
            ctypes.c_double(coefficient),
            pauli_string.encode('utf-8')
        )
        if ret != 0:
            raise QuantumError("Failed to add Hamiltonian term")


# ============================================================================
# QAOA CLASS
# ============================================================================

class QAOA:
    """
    Quantum Approximate Optimization Algorithm

    Solves combinatorial optimization problems encoded as Ising models.
    Supports MaxCut, graph partitioning, and custom Ising problems.

    Example:
        qaoa = QAOA(num_qubits=5, num_layers=3)
        result = qaoa.solve_maxcut(edges=[(0,1), (1,2), (2,3), (3,4), (4,0)])
        print(f"Best cut: {result['best_bitstring']}, cost: {result['best_cost']}")
    """

    def __init__(self, num_qubits: int, num_layers: int = 3):
        """
        Initialize QAOA solver

        Args:
            num_qubits: Number of qubits/vertices
            num_layers: QAOA circuit depth (p parameter)
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self._model = None
        self._solver = None
        self._graph = None

    def __del__(self):
        """Clean up C resources"""
        if hasattr(self, '_solver') and self._solver:
            _lib.qaoa_solver_free(self._solver)
        if hasattr(self, '_model') and self._model:
            _lib.ising_model_free(self._model)
        if hasattr(self, '_graph') and self._graph:
            _lib.graph_free(self._graph)

    def solve_maxcut(self, edges: List[Tuple[int, int]],
                     weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Solve MaxCut problem on a graph

        Args:
            edges: List of (u, v) edge tuples
            weights: Optional edge weights (default: all 1.0)

        Returns:
            Dict with 'best_bitstring', 'best_cost', 'expectation', 'converged'
        """
        if weights is None:
            weights = [1.0] * len(edges)

        # Create graph
        if self._graph:
            _lib.graph_free(self._graph)

        self._graph = _lib.graph_create(
            ctypes.c_size_t(self.num_qubits),
            ctypes.c_size_t(len(edges))
        )
        if not self._graph:
            raise QuantumError("Failed to create graph")

        for i, ((u, v), w) in enumerate(zip(edges, weights)):
            _lib.graph_add_edge(
                self._graph,
                ctypes.c_size_t(i),
                ctypes.c_int(u), ctypes.c_int(v),
                ctypes.c_double(w)
            )

        # Encode as Ising model
        if self._model:
            _lib.ising_model_free(self._model)

        self._model = _lib.ising_encode_maxcut(self._graph)
        if not self._model:
            raise QuantumError("Failed to encode MaxCut as Ising model")

        return self._solve()

    def solve_ising(self, J: np.ndarray, h: np.ndarray) -> Dict[str, Any]:
        """
        Solve general Ising model: H = Σ J_ij σ_i σ_j + Σ h_i σ_i

        Args:
            J: Coupling matrix (n x n)
            h: Field vector (n,)

        Returns:
            Dict with 'best_bitstring', 'best_cost', 'expectation', 'converged'
        """
        if self._model:
            _lib.ising_model_free(self._model)

        self._model = _lib.ising_model_create(ctypes.c_size_t(self.num_qubits))
        if not self._model:
            raise QuantumError("Failed to create Ising model")

        # Set couplings
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                if J[i, j] != 0:
                    _lib.ising_model_set_coupling(
                        self._model,
                        ctypes.c_size_t(i), ctypes.c_size_t(j),
                        ctypes.c_double(J[i, j])
                    )

        # Set fields
        for i in range(self.num_qubits):
            if h[i] != 0:
                _lib.ising_model_set_field(
                    self._model,
                    ctypes.c_size_t(i),
                    ctypes.c_double(h[i])
                )

        return self._solve()

    def _solve(self) -> Dict[str, Any]:
        """Internal solve method"""
        if self._solver:
            _lib.qaoa_solver_free(self._solver)

        self._solver = _lib.qaoa_solver_create(
            self._model,
            ctypes.c_size_t(self.num_layers)
        )
        if not self._solver:
            raise QuantumError("Failed to create QAOA solver")

        result = _lib.qaoa_solve(self._solver)

        # Extract parameters
        gamma = []
        beta = []
        if result.optimal_gamma and result.optimal_beta:
            for i in range(result.num_layers):
                gamma.append(result.optimal_gamma[i])
                beta.append(result.optimal_beta[i])

        return {
            'expectation': result.expectation,
            'optimal_gamma': np.array(gamma),
            'optimal_beta': np.array(beta),
            'best_bitstring': result.best_bitstring,
            'best_cost': result.best_cost,
            'converged': bool(result.converged),
        }

    def compute_expectation(self, gamma: np.ndarray, beta: np.ndarray) -> float:
        """
        Compute cost expectation for given QAOA parameters

        Args:
            gamma: Phase separation angles
            beta: Mixer angles

        Returns:
            Expected cost value
        """
        if not self._solver:
            raise QuantumError("Must call solve_*() first")

        c_gamma = (ctypes.c_double * len(gamma))(*gamma)
        c_beta = (ctypes.c_double * len(beta))(*beta)

        return _lib.qaoa_compute_expectation(
            self._solver,
            c_gamma,
            c_beta,
            ctypes.c_size_t(len(gamma))
        )

    def evaluate_bitstring(self, bitstring: int) -> float:
        """
        Evaluate the cost of a specific bitstring

        Args:
            bitstring: Integer encoding of the solution

        Returns:
            Cost value
        """
        if not self._model:
            raise QuantumError("No Ising model defined")

        return _lib.ising_model_evaluate(self._model, ctypes.c_uint64(bitstring))


# ============================================================================
# GROVER CLASS
# ============================================================================

class Grover:
    """
    Grover's quantum search algorithm

    Finds marked items in unstructured search with quadratic speedup.

    Example:
        grover = Grover(num_qubits=10)
        result = grover.search(marked_state=42)
        print(f"Found: {result['found_state']}, success: {result['success']}")
    """

    def __init__(self, num_qubits: int):
        """
        Initialize Grover search

        Args:
            num_qubits: Search space size = 2^num_qubits
        """
        self.num_qubits = num_qubits
        self._state = QuantumState(num_qubits)

    @property
    def optimal_iterations(self) -> int:
        """Get optimal number of Grover iterations"""
        return int(_lib.grover_optimal_iterations(ctypes.c_size_t(self.num_qubits)))

    def search(self, marked_state: int,
               num_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Search for marked state

        Args:
            marked_state: The state to find (0 to 2^n - 1)
            num_iterations: Number of iterations (default: optimal)

        Returns:
            Dict with 'found_state', 'success', 'probability', 'iterations'
        """
        if marked_state < 0 or marked_state >= (1 << self.num_qubits):
            raise ValueError(f"Marked state must be in [0, {(1 << self.num_qubits) - 1}]")

        # Initialize to uniform superposition
        self._state.reset()
        for i in range(self.num_qubits):
            self._state.h(i)

        # Set up config
        config = CGroverConfig()
        config.marked_state = ctypes.c_uint64(marked_state)
        config.num_qubits = ctypes.c_size_t(self.num_qubits)
        config.use_optimal_iterations = 1 if num_iterations is None else 0
        config.num_iterations = ctypes.c_size_t(
            num_iterations if num_iterations else self.optimal_iterations
        )

        # Run search
        result = _lib.grover_search(
            ctypes.byref(self._state._state),
            ctypes.byref(config),
            None  # entropy context
        )

        return {
            'found_state': result.found_state,
            'success': bool(result.success),
            'probability': result.probability,
            'iterations_used': result.iterations_used,
        }

    def step(self, marked_state: int) -> None:
        """
        Perform a single Grover iteration

        Args:
            marked_state: The marked state
        """
        _lib.grover_iteration(
            ctypes.byref(self._state._state),
            ctypes.c_uint64(marked_state)
        )

    def oracle(self, marked_state: int) -> None:
        """
        Apply Grover oracle (phase flip on marked state)

        Args:
            marked_state: State to mark with -1 phase
        """
        _lib.grover_oracle(
            ctypes.byref(self._state._state),
            ctypes.c_uint64(marked_state)
        )

    def diffusion(self) -> None:
        """Apply Grover diffusion operator (inversion about mean)"""
        _lib.grover_diffusion(ctypes.byref(self._state._state))

    @property
    def state(self) -> QuantumState:
        """Access the underlying quantum state"""
        return self._state

    def probabilities(self) -> np.ndarray:
        """Get measurement probabilities for all states"""
        return self._state.probabilities()


# ============================================================================
# BELL TEST CLASS
# ============================================================================

class BellTest:
    """
    Bell inequality testing for quantum correlations

    Tests CHSH inequality to verify quantum entanglement.
    Classical bound: S ≤ 2, Quantum bound: S ≤ 2√2 ≈ 2.828

    Example:
        result = BellTest.chsh_test(state, qubit_a=0, qubit_b=1)
        print(f"CHSH S = {result['chsh']}, violates classical: {result['violates']}")
    """

    # Bell state types
    PHI_PLUS = 0   # |Φ+⟩ = (|00⟩ + |11⟩)/√2
    PHI_MINUS = 1  # |Φ-⟩ = (|00⟩ - |11⟩)/√2
    PSI_PLUS = 2   # |Ψ+⟩ = (|01⟩ + |10⟩)/√2
    PSI_MINUS = 3  # |Ψ-⟩ = (|01⟩ - |10⟩)/√2

    @staticmethod
    def create_bell_state(state: QuantumState, qubit_a: int, qubit_b: int,
                          bell_type: int = 0) -> None:
        """
        Create a Bell state on two qubits

        Args:
            state: Quantum state to modify
            qubit_a: First qubit
            qubit_b: Second qubit
            bell_type: PHI_PLUS, PHI_MINUS, PSI_PLUS, or PSI_MINUS
        """
        ret = _lib.create_bell_state(
            ctypes.byref(state._state),
            ctypes.c_int(qubit_a),
            ctypes.c_int(qubit_b),
            ctypes.c_int(bell_type)
        )
        if ret != 0:
            raise QuantumError("Failed to create Bell state")

    @staticmethod
    def chsh_test(state: QuantumState, qubit_a: int, qubit_b: int,
                  num_measurements: int = 10000) -> Dict[str, Any]:
        """
        Perform CHSH Bell test

        Uses optimal measurement angles: a=0, a'=π/2, b=π/4, b'=3π/4

        Args:
            state: Entangled quantum state
            qubit_a: Alice's qubit
            qubit_b: Bob's qubit
            num_measurements: Statistics per correlation (default: 10000)

        Returns:
            Dict with 'chsh', 'correlations', 'violates_classical', 'variance'
        """
        # Get optimal settings
        settings = CBellMeasurementSettings()
        _lib.bell_get_optimal_settings(ctypes.byref(settings))

        # Run test
        result = _lib.bell_test_chsh(
            ctypes.byref(state._state),
            ctypes.c_int(qubit_a),
            ctypes.c_int(qubit_b),
            ctypes.byref(settings),
            ctypes.c_size_t(num_measurements),
            None  # entropy context
        )

        correlations = [result.correlations[i] for i in range(4)]

        return {
            'chsh': result.chsh_parameter,
            'correlations': {
                'E(a,b)': correlations[0],
                'E(a,b\')': correlations[1],
                'E(a\',b)': correlations[2],
                'E(a\',b\')': correlations[3],
            },
            'violates_classical': bool(result.violates_classical),
            'num_measurements': result.num_measurements,
            'variance': result.variance,
        }

    @staticmethod
    def quick_test(num_qubits: int = 2, num_measurements: int = 1000) -> Dict[str, Any]:
        """
        Quick Bell test with automatic state preparation

        Creates a Bell state and tests CHSH inequality.

        Args:
            num_qubits: Number of qubits (default: 2)
            num_measurements: Measurements per correlation

        Returns:
            CHSH test results
        """
        state = QuantumState(num_qubits)
        BellTest.create_bell_state(state, 0, 1, BellTest.PHI_PLUS)
        return BellTest.chsh_test(state, 0, 1, num_measurements)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_vqe_h2(bond_distance: float = 0.74, num_layers: int = 2) -> Dict[str, Any]:
    """
    Quick VQE run for H2 molecule

    Args:
        bond_distance: H-H distance in Angstroms
        num_layers: Ansatz depth

    Returns:
        VQE result dictionary
    """
    vqe = VQE(num_qubits=4, num_layers=num_layers)
    return vqe.solve_h2(bond_distance)


def run_qaoa_maxcut(edges: List[Tuple[int, int]],
                    num_layers: int = 3) -> Dict[str, Any]:
    """
    Quick QAOA run for MaxCut

    Args:
        edges: Graph edges as (u, v) tuples
        num_layers: QAOA depth

    Returns:
        QAOA result dictionary
    """
    num_vertices = max(max(u, v) for u, v in edges) + 1
    qaoa = QAOA(num_qubits=num_vertices, num_layers=num_layers)
    return qaoa.solve_maxcut(edges)


def run_grover(num_qubits: int, marked_state: int) -> Dict[str, Any]:
    """
    Quick Grover search

    Args:
        num_qubits: Search space = 2^num_qubits
        marked_state: Target to find

    Returns:
        Grover result dictionary
    """
    grover = Grover(num_qubits)
    return grover.search(marked_state)


def run_bell_test(num_measurements: int = 10000) -> Dict[str, Any]:
    """
    Quick Bell test with optimal state

    Args:
        num_measurements: Statistics per correlation

    Returns:
        Bell test result dictionary
    """
    return BellTest.quick_test(num_measurements=num_measurements)
