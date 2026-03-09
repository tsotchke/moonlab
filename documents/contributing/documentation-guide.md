# Documentation Guide

Standards and best practices for writing Moonlab documentation.

## Overview

Moonlab documentation targets researchers and professional developers in quantum computing. Content should be academically rigorous while remaining accessible to those learning quantum simulation.

## Documentation Structure

### File Organization

```
docs/
├── index.md                    # Landing page
├── quickstart.md               # 5-minute getting started
├── installation.md             # Platform-specific install
│
├── concepts/                   # Theoretical background
│   ├── quantum-computing-basics.md
│   ├── state-vector-simulation.md
│   └── ...
│
├── getting-started/            # Beginner track
│   ├── prerequisites.md
│   ├── first-simulation.md
│   └── ...
│
├── tutorials/                  # Step-by-step learning
│   ├── 01-hello-quantum.md
│   ├── 02-quantum-gates-tour.md
│   └── ...
│
├── guides/                     # Task-oriented how-to
│   ├── gpu-acceleration.md
│   ├── performance-tuning.md
│   └── ...
│
├── api/                        # API reference
│   ├── c/                      # C API
│   ├── python/                 # Python API
│   └── ...
│
├── algorithms/                 # Algorithm deep-dives
├── architecture/               # Internal design
├── performance/                # Benchmarks
├── examples/                   # Code examples
├── contributing/               # Contributor docs
└── reference/                  # Quick reference
```

### Page Types

| Type | Purpose | Audience | Example |
|------|---------|----------|---------|
| **Concept** | Theoretical background | All levels | `quantum-gates.md` |
| **Tutorial** | Step-by-step learning | Beginners | `01-hello-quantum.md` |
| **Guide** | Task-oriented how-to | Intermediate | `gpu-acceleration.md` |
| **API Reference** | Complete API docs | All levels | `api/c/gates.md` |
| **Example** | Working code samples | All levels | `examples/bell-state.md` |
| **Architecture** | Internal design | Contributors | `state-vector-engine.md` |

## Writing Style

### Tone and Voice

- **Academic but accessible**: Rigorous content without unnecessary jargon
- **Professional**: Publication-quality prose
- **Direct**: Active voice, imperative mood for instructions
- **Precise**: Exact terminology, no ambiguity

```markdown
<!-- GOOD: Direct, precise -->
Apply the Hadamard gate to qubit 0 to create superposition:

<!-- BAD: Passive, vague -->
Superposition can be created by having the Hadamard gate applied.
```

### Audience Assumptions

**Assumed knowledge**:
- Linear algebra (vectors, matrices, tensor products)
- Basic probability theory
- Programming experience (C or Python)
- Familiarity with complex numbers

**Not assumed**:
- Quantum mechanics beyond basics
- Specific quantum computing background
- Hardware implementation details

## Formatting Standards

### Headings

Use ATX-style headings with proper hierarchy:

```markdown
# Page Title (H1 - one per page)

## Major Section (H2)

### Subsection (H3)

#### Minor Topic (H4 - use sparingly)
```

### Mathematical Notation

Use LaTeX for all mathematical expressions:

**Inline math**: Use `$...$`
```markdown
A qubit state $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$ satisfies $|\alpha|^2 + |\beta|^2 = 1$.
```

**Display math**: Use `$$...$$`
```markdown
The Hadamard gate matrix:

$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$
```

**Common notation**:

| Concept | Notation | LaTeX |
|---------|----------|-------|
| Ket | $\|0\rangle$ | `$\|0\rangle$` |
| Bra | $\langle 0\|$ | `$\langle 0\|$` |
| Inner product | $\langle\psi\|\phi\rangle$ | `$\langle\psi\|\phi\rangle$` |
| Tensor product | $\|\psi\rangle \otimes \|\phi\rangle$ | `$\|\psi\rangle \otimes \|\phi\rangle$` |
| Norm | $\|\|\psi\rangle\|$ | `$\|\|\psi\rangle\|$` |
| Complex conjugate | $\alpha^*$ | `$\alpha^*$` |
| Hermitian conjugate | $U^\dagger$ | `$U^\dagger$` |
| Trace | $\text{Tr}(\rho)$ | `$\text{Tr}(\rho)$` |

### Code Blocks

**Inline code**: Use backticks for function names, variables, file paths:
```markdown
Call `quantum_state_init()` to create a new state.
```

**Code blocks**: Use fenced blocks with language identifier:

````markdown
```c
qs_error_t err = quantum_state_init(&state, 10);
if (err != QS_SUCCESS) {
    fprintf(stderr, "Failed to initialize state\n");
    return 1;
}
```
````

**Supported languages**: `c`, `python`, `rust`, `javascript`, `bash`, `json`, `yaml`

### Tables

Use tables for comparisons and reference data:

```markdown
| Gate | Matrix | Effect |
|------|--------|--------|
| X | $\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ | Bit flip |
| Z | $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ | Phase flip |
| H | $\frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ | Superposition |
```

### Diagrams

Include ASCII diagrams for circuits and architecture:

```markdown
Quantum teleportation circuit:

```
       ┌───┐      ░ ┌─┐
q0: |ψ⟩┤ H ├──■───░─┤M├─────────
       └───┘┌─┴─┐ ░ └╥┘┌─┐
q1: |0⟩─────┤ X ├─░──╫─┤M├──────
            └───┘ ░  ║ └╥┘ ┌───┐
q2: |0⟩───────────░──╫──╫──┤ ? ├
                  ░  ║  ║  └───┘
c0: ═════════════════╩══╬═══════
c1: ════════════════════╩═══════
```
```

For complex diagrams, reference SVG files in `docs/_assets/images/`.

### Admonitions

Use blockquotes with bold prefixes for special callouts:

```markdown
> **Note**: Additional information that may be helpful.

> **Warning**: Important caution about potential issues.

> **Tip**: Helpful suggestion for better results.

> **Example**: Illustrative case demonstrating a concept.
```

## API Documentation

### Function Documentation

Document every public function with:

1. **Brief description**: One-line summary
2. **Detailed description**: Full explanation, mathematical background
3. **Parameters**: Type and purpose of each parameter
4. **Return value**: What is returned and possible error codes
5. **Example**: Working code sample
6. **Notes/Warnings**: Edge cases, performance considerations

**Template**:

````markdown
## `function_name`

Brief one-line description.

```c
return_type function_name(param_type1 param1, param_type2 param2);
```

Detailed description with mathematical notation if applicable.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `param1` | `param_type1` | Description of first parameter |
| `param2` | `param_type2` | Description of second parameter |

### Returns

Description of return value. For functions returning error codes:

| Value | Meaning |
|-------|---------|
| `QS_SUCCESS` | Operation completed successfully |
| `QS_ERROR_INVALID_QUBIT` | Qubit index out of range |

### Example

```c
quantum_state_t state;
qs_error_t err = function_name(&state, 10);
if (err != QS_SUCCESS) {
    // Handle error
}
```

### Notes

- Performance considerations
- Thread safety
- Memory management

### See Also

- [`related_function`](related.md)
````

### Type Documentation

Document structs and enums completely:

````markdown
## `quantum_state_t`

Represents a pure quantum state |ψ⟩ = Σ αᵢ|i⟩.

```c
typedef struct {
    size_t num_qubits;
    size_t state_dim;
    complex_t *amplitudes;
    double entanglement_entropy;
    double purity;
} quantum_state_t;
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `num_qubits` | `size_t` | Number of qubits in the system |
| `state_dim` | `size_t` | Dimension of state space (2^n) |
| `amplitudes` | `complex_t*` | State vector coefficients |
| `entanglement_entropy` | `double` | Von Neumann entropy |
| `purity` | `double` | Tr(ρ²), 1.0 for pure states |
````

## Tutorial Writing

### Structure

Every tutorial should follow this structure:

1. **Title and overview**: What will be learned
2. **Prerequisites**: Required knowledge and setup
3. **Learning objectives**: Specific skills gained
4. **Step-by-step instructions**: Numbered, clear steps
5. **Complete code**: Full working example
6. **Explanation**: Why each part works
7. **Exercises**: Practice problems
8. **Next steps**: Where to go from here

### Example Tutorial Section

````markdown
## Step 3: Create Entanglement

Now we'll create a Bell state using CNOT:

```c
// Apply CNOT with qubit 0 as control, qubit 1 as target
qs_error_t err = gate_cnot(&state, 0, 1);
if (err != QS_SUCCESS) {
    fprintf(stderr, "CNOT failed\n");
    return 1;
}
```

The CNOT gate flips the target qubit when the control qubit is |1⟩.
Starting from |+0⟩ = (|0⟩ + |1⟩)|0⟩/√2, we get:

$$\text{CNOT}_{01} \cdot \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle) = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

This is the Bell state |Φ⁺⟩, a maximally entangled state.
````

## Cross-Referencing

### Internal Links

Use relative paths for internal documentation links:

```markdown
See the [Quantum Gates](../concepts/quantum-gates.md) documentation.

For API details, see [`quantum_state_init`](../api/c/quantum-state.md#quantum_state_init).
```

### Section Links

Link to specific sections using anchors:

```markdown
See [Error Handling](#error-handling) below.

Refer to [SIMD Optimization](../architecture/gate-implementation.md#simd-optimization).
```

### See Also Sections

End each document with related resources:

```markdown
## See Also

- [Quantum Gates Reference](../reference/gate-reference.md)
- [State Vector Engine](../architecture/state-vector-engine.md)
- [API: gates.h](../api/c/gates.md)
```

## Building Documentation

### Local Preview

```bash
# Install MkDocs (if using)
pip install mkdocs mkdocs-material

# Serve locally
mkdocs serve

# Open http://localhost:8000
```

### Building Static Site

```bash
# Build HTML
mkdocs build

# Output in site/ directory
```

### Documentation CI

Documentation is built automatically on pull requests:

```yaml
# .github/workflows/docs.yml
name: Documentation
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build docs
        run: mkdocs build --strict
```

## Review Checklist

Before submitting documentation:

- [ ] **Accuracy**: Code examples compile and run correctly
- [ ] **Completeness**: All parameters, return values documented
- [ ] **Consistency**: Follows existing style and terminology
- [ ] **Links**: All internal links work
- [ ] **Math**: LaTeX renders correctly
- [ ] **Code**: Syntax highlighting works, examples are tested
- [ ] **Spelling**: No typos or grammatical errors
- [ ] **See Also**: Related documents linked

## Common Mistakes

### Avoid These

```markdown
<!-- BAD: Vague descriptions -->
This function does some stuff with the quantum state.

<!-- GOOD: Specific descriptions -->
Applies the Hadamard gate to the specified qubit, creating an equal superposition of |0⟩ and |1⟩.
```

```markdown
<!-- BAD: Missing error handling in examples -->
quantum_state_init(&state, 10);
gate_hadamard(&state, 0);

<!-- GOOD: Complete examples with error handling -->
qs_error_t err = quantum_state_init(&state, 10);
if (err != QS_SUCCESS) {
    fprintf(stderr, "Init failed: %d\n", err);
    return 1;
}
```

```markdown
<!-- BAD: Undefined abbreviations -->
The MPS representation uses SVD for truncation.

<!-- GOOD: Define abbreviations first -->
Matrix Product States (MPS) use Singular Value Decomposition (SVD) for truncation.
```

## File Naming

- Use lowercase with hyphens: `quantum-gates.md`
- Match content hierarchy: `tutorials/01-hello-quantum.md`
- Be descriptive: `distributed-architecture.md` not `dist.md`

## Versioning

- Document version-specific features clearly
- Use admonitions for deprecated features
- Maintain changelog in `CHANGELOG.md`

```markdown
> **Note**: This feature requires Moonlab 2.0 or later.

> **Deprecated**: Use `quantum_state_init_v2()` instead. Will be removed in version 3.0.
```

## See Also

- [Development Setup](development-setup.md) - Build environment
- [Code Style](code-style.md) - Code formatting
- [Testing Guide](testing-guide.md) - Testing practices
