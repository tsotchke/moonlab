# Moonlab: Mathematical Foundations and Error Bounds

This document gives the analytic underpinnings of each major
algorithm in Moonlab, including the error bounds that motivate the
test tolerances and benchmark thresholds. It is a companion to
`ARCHITECTURE.md` and is meant for readers who want to understand
*why* the numerical choices in the library look the way they do.

## 1. The Matrix Sign Function and the Projector

The filled-band projector for a gapped quadratic Hamiltonian
@f$\hat H@f$ is
@f[
  \hat P \;=\; \tfrac12\bigl(\mathbb 1 - \operatorname{sign}\hat H\bigr),
@f]
where @f$\operatorname{sign}@f$ acts on the eigenvalues (±1 above and
below the gap). Moonlab computes @f$\operatorname{sign}\hat H@f$
through the **Newton-Schulz iteration**
@f[
  Y_{k+1} \;=\; \tfrac12\,Y_k\bigl(3\mathbb 1 - Y_k^{2}\bigr),
  \qquad Y_0 = \hat H / B,
@f]
with @f$B > \lVert\hat H\rVert_2@f$ so the initial spectrum is
strictly inside @f$(-1, 1)\setminus\{0\}@f$.

**Convergence.** Higham, *Functions of Matrices*, Theorem 5.6
(Newton's sign iteration) establishes that, provided
@f$\operatorname{spec}(Y_0)@f$ avoids the imaginary axis, the
iteration converges *quadratically*:
@f[
  \lVert Y_{k+1} - \operatorname{sign}Y_\infty \rVert
  \;\lesssim\; \tfrac12\,\kappa\,\lVert Y_k - \operatorname{sign}Y_\infty\rVert^{2},
@f]
with @f$\kappa@f$ set by the spectral radius.  For our QWZ models
with mass @f$|m|\sim 1@f$, the spectral gap is
@f$\Delta \gtrsim 1@f$ after rescaling by @f$B@f$, and the iteration
reaches
@f$\lVert Y^2 - \mathbb 1 \rVert_F^2 < 10^{-12}@f$ in roughly
@f$\log_2\log_2(\kappa / \epsilon) \approx 6@f$ iterations.  This is
the `tol = 1e-12` exit condition in
`src/algorithms/topology_realspace/chern_marker.c`.

**Ungapped case.** When @f$\Delta \to 0@f$ the Schulz iteration
degrades: the factor @f$3 - Y^2@f$ approaches zero near eigenvalues
sitting on the imaginary axis in the complex-plane picture.  Users
that evaluate the marker at a phase transition where the gap closes
must either widen @f$B@f$ or switch to a smoother projector
construction (KPM with Jackson kernel below).

## 2. KPM / Chebyshev Expansion of the Sign Function

At large lattice sizes the dense Schulz iteration is unaffordable
(memory @f$O(N^2)@f$ for @f$N = L^2\,\mathrm{orbs}@f$).
`chern_kpm.c` uses the kernel polynomial method (KPM) of Weisse,
Wellein, Alvermann and Fehske to apply @f$\hat P@f$ to vectors
without ever materialising it.

Let @f$\hat{\tilde H} = (\hat H - a\mathbb 1)/b@f$ with
@f$b > (E_{\max}-E_{\min})/2@f$ so
@f$\operatorname{spec}(\hat{\tilde H}) \subset (-1, 1)@f$.  The
Chebyshev expansion of the sign function on @f$[-1, 1]@f$ is
@f[
  \operatorname{sign}\varepsilon \;\sim\; \sum_{n=0}^{\infty} c_n\,T_n(\varepsilon),
  \qquad
  c_n \;=\; \frac{2}{\pi}\!\int_{-1}^{1}\! \frac{\operatorname{sign}\varepsilon\,T_n(\varepsilon)}{\sqrt{1-\varepsilon^2}}\,d\varepsilon
        \;=\; \begin{cases} 0 & n\ \text{even} \\
                            \dfrac{4}{\pi n}\,\sin\!\frac{n\pi}{2} & n\ \text{odd}.
                            \end{cases}
@f]
Truncating at @f$N_c@f$ terms Gibbs-oscillates near
@f$\varepsilon = 0@f$.  The **Jackson kernel**
@f[
  g_n^{(J)} \;=\; \frac{(N_c - n + 1)\cos\frac{n\pi}{N_c+1} + \sin\frac{n\pi}{N_c+1}\cot\frac{\pi}{N_c+1}}{N_c+1}
@f]
smooths the truncation: the Jackson-regularised approximant
@f$\operatorname{sign}_{\mathrm{K}}(\varepsilon) = \sum_{n<N_c} g_n^{(J)}\,c_n\,T_n(\varepsilon)@f$
is bounded between @f$\pm 1@f$, non-negative in a broadened strip
around the original sign, and converges uniformly on any compact
subset of @f$[-1, 1]\setminus(-\delta, \delta)@f$ at rate
@f[
  \sup_{|\varepsilon|\ge\delta}
  \bigl|\operatorname{sign}(\varepsilon) - \operatorname{sign}_{\mathrm{K}}(\varepsilon)\bigr|
  \;=\; O(1/(N_c\,\delta))
@f]
(Weisse et al., §V).  For QWZ models with physical band gap
@f$\Delta_{\mathrm{phys}} \gtrsim 1@f$ and rescale
@f$b \lesssim 3@f$, the effective gap in the rescaled spectrum is
@f$\delta \sim 1/3@f$, so the leading KPM residual at
@f$N_c = 100@f$ is bounded by a few times @f$10^{-2}@f$, below the
finite-size correction to the bulk Bianco-Resta marker on the
@f$L \sim 20@f$ lattices used by the dense reference.  This is why
`unit_chern_kpm` can reliably compare KPM vs dense at
@f$|c_{\mathrm{dense}} - c_{\mathrm{KPM}}| < 10^{-3}@f$: both
exceed KPM truncation error and finite-size error at those settings.

## 3. Bianco-Resta Local Chern Marker

For a gapped 2D insulator on a square lattice with unit-cell area
@f$A_{\mathrm{uc}}@f$, Bianco and Resta (2011, Eq. 2) define the
*local Chern marker* at site @f$\mathbf r@f$:
@f[
  c(\mathbf r) \;=\; -\frac{4\pi}{A_{\mathrm{uc}}}\,\operatorname{Im}
  \sum_\alpha \langle\mathbf r,\alpha|\,\hat P\,\hat X\,\hat Q\,\hat Y\,\hat P\,|\mathbf r,\alpha\rangle,
@f]
where @f$\hat X, \hat Y@f$ are the on-site position operators and
@f$\hat Q = \mathbb 1 - \hat P@f$.  Summed over the lattice,
@f$\sum_{\mathbf r} c(\mathbf r)@f$ is the Bianco-Resta ground-state
expectation of @f$-4\pi\,\operatorname{Im}\hat P[\hat X,\hat Y]\hat P / A_{\mathrm{uc}}@f$.
Its torus average converges, by the Bloch-space identity
@f$\hat P[\hat X,\hat Y]\hat P = i\,\hat P\,\partial_k^{x}\hat P\,\partial_k^{y}\hat P@f$
(modulo boundary terms), to the integer Chern number of the filled
band.  The factor @f$-4\pi/A_{\mathrm{uc}}@f$ fixes that
normalisation; on Moonlab's unit-lattice convention @f$A_{\mathrm{uc}} = 1@f$.

**Finite-size corrections** scale as @f$O(e^{-L/\xi})@f$ in the bulk
(exponential in the ratio of linear size to correlation length),
plus an @f$O(1/L)@f$ envelope from the truncation of the (infinite)
position operators.  At @f$L = 14@f$ on QWZ with @f$m = -1@f$ the
measured centre-of-lattice marker is @f$+1.000@f$ to four decimal
places, well within the test tolerance of @f$0.2@f$.

## 4. Fukui-Hatsugai-Suzuki Quantisation

For the momentum-space Chern number, `qgt_berry_grid` uses the FHS
link-variable construction: define
@f$U_\mu(\mathbf k) = \langle u(\mathbf k)|u(\mathbf k+\delta_\mu)\rangle / |\cdot|@f$
and the plaquette flux
@f[
  F_{xy}(\mathbf k) \;=\; \operatorname{arg}
  \bigl[U_x(\mathbf k)\,U_y(\mathbf k+\delta_x)\,U_x(\mathbf k+\delta_y)^{-1}\,U_y(\mathbf k)^{-1}\bigr]_{(-\pi,\pi]}.
@f]
**Theorem (Fukui-Hatsugai-Suzuki 2005):** at any finite grid size
@f$N@f$, provided the band is gapped at every grid point,
@f[
  C \;\equiv\; \frac{1}{2\pi}\sum_{\mathbf k} F_{xy}(\mathbf k) \;\in\; \mathbb Z,
@f]
with no branch-cut ambiguity.  The integer @f$C@f$ equals the true
Chern number once @f$N@f$ is large enough for the plaquette
curvature to be resolvable (no @f$2\pi@f$ jumps between adjacent
plaquettes).  This is why Moonlab's `test_qgt` insists on *exactly*
integer output at @f$N = 32@f$ via `lround(c)`; the deviation from
an integer is a sanity check that no gauge discontinuity has slipped
through the eigenvector-selection step.

**Eigenvector-gauge robustness.**  The two-band lower-eigenvector
formulas
@f$\mathbf u_A = (h_x - ih_y,\, -(h_z+|\mathbf h|))^{\mathrm T}@f$ and
@f$\mathbf u_B = (h_z-|\mathbf h|,\, h_x+ih_y)^{\mathrm T}@f$ each
have a single zero on the @f$|\mathbf h|@f$-sphere (at
@f$\mathbf h\parallel\pm\hat z@f$ respectively).  Switching between
them across a grid face introduces a @f$\pi@f$ phase jump that
masquerades as a real plaquette flux; we pick whichever formula has
larger pre-normalisation 2-norm at the current @f$\mathbf k@f$,
which guarantees we never sit within @f$\sqrt\epsilon@f$ of the
zero of the chosen formula.  Exact integer Chern output at
@f$N = 32@f$ on the full QWZ phase diagram is the empirical proof
that the gauge is stable in practice.

## 5. Mezzadri Haar Sampling and QV Statistics

The Quantum Volume protocol (Cross et al. 2019) requires
*Haar-random* elements of @f$U(4)@f$.  Mezzadri (2007, Theorem 2)
shows that the naive recipe "sample a @f$4 \times 4@f$ i.i.d.
complex-Gaussian matrix @f$Z@f$, take its QR factorisation, and
return @f$Q@f$" yields a *non-uniform* distribution on @f$U(4)@f$
because Gram-Schmidt is not continuous with respect to the Haar
measure.  The fix is to rotate each column of @f$Q@f$ by the phase
@f$\overline{R_{jj}/|R_{jj}|}@f$, which drags the triangular-factor
diagonal onto the positive real axis.  `src/applications/quantum_volume.c`
implements exactly this construction.

**Heavy-output statistics.**  For a Haar-random
@f$U \in U(2^d)@f$, the distribution of the output amplitudes
@f$|\langle x|U|0\rangle|^2@f$ is the Porter-Thomas distribution
(an exponential on @f$[0, \infty)@f$ with mean @f$2^{-d}@f$ in the
large-@f$d@f$ limit).  A direct integration yields the asymptotic
mean heavy-output probability
@f[
  \bar{\mathrm{hop}}_{\mathrm{PT}} \;=\; (1 + \ln 2)/2 \;\approx\; 0.8467.
@f]
The QV protocol passes a width @f$d@f$ iff the one-sided 97.5 %%
lower confidence bound on the sample mean exceeds @f$2/3@f$ (a
value derived from the classical depolarising baseline
@f$\tfrac12\cdot 1 + \tfrac12\cdot 2^{-d}\to\tfrac12@f$ combined
with noise margins; see Cross et al. §IV).  On Moonlab's noiseless
simulator, the variance of the heavy-output probability shrinks
like @f$O(1/N_{\mathrm{trials}})@f$ and is dominated by Haar-sample
variability, *not* by shot noise (we evaluate the heavy outputs
exactly from the full state vector).  The normal-approximation
confidence bound is therefore conservative; a bootstrap would give
similar or slightly tighter intervals.

## 6. Zak Phase / SSH Winding Number Convention

For the SSH Hamiltonian
@f$H(k) = (t_1 + t_2\cos k)\sigma_x + (t_2\sin k)\sigma_y@f$, the
lower-band Bloch state winds in the @f$(\sigma_x,\sigma_y)@f$-plane;
the Zak phase of the lower band is
@f[
  \gamma_{\mathrm{Zak}} \;=\; i\oint_{\mathrm{BZ}} \langle u_-(k)|\partial_k u_-(k)\rangle\,dk
  \;=\; \pi\,W,
@f]
where @f$W \in \mathbb Z@f$ is the winding number (equal to
@f$0@f$ for @f$|t_2| < |t_1|@f$ and @f$\pm 1@f$ for
@f$|t_2| > |t_1|@f$; the sign depends on orientation of the gauge
choice).  Our implementation in `qgt_winding_1d` computes
@f$\gamma@f$ discretely and returns
@f$W = -\gamma_{\mathrm{Zak}}/\pi@f$.  The global sign (@f$-@f$)
fixes the convention @f$W = +1@f$ in the topological phase
@f$|t_2| > |t_1|@f$, which is the standard choice in Asbóth,
Oroszlány and Pályi's *A Short Course on Topological Insulators*
(Springer 2016) §1.5.  Moonlab's tests assert
@f$W = +1@f$ for @f$t_2 > t_1 > 0@f$ and @f$W = 0@f$ otherwise.

## 7. QPE Precision Guarantee

For an eigenstate @f$|\psi\rangle@f$ with @f$U|\psi\rangle = e^{2\pi i\varphi}|\psi\rangle@f$
and an @f$m@f$-qubit ancilla register, the QPE measurement outcome
@f$y \in \{0, \ldots, 2^m - 1\}@f$ satisfies
@f[
  \Pr\!\left[\bigl|\tilde\varphi - \varphi\bigr| \le 2^{-m}\right]
  \;\ge\; \frac{4}{\pi^2} \;\approx\; 0.405
@f]
with @f$\tilde\varphi = y/2^m@f$.  The bound is tight and is the
reason `test_qpe` uses @f$m \ge 5@f$ precision qubits to drive the
total confidence above 95 %% with a handful of repetitions
(see Cleve-Ekert-Macchiavello-Mosca 1998, §III).

## 8. Gate-Fusion Correctness

Fusing consecutive single-qubit gates on the same qubit is exact
(no approximation introduced): if @f$U_1, \ldots, U_k@f$ all act on
the same qubit @f$q@f$ with trivial identity factors elsewhere,
then @f$U_k\cdots U_1@f$ is also a single-qubit operator on @f$q@f$.
The fused circuit has identical unitary action as the original
circuit on every state vector.  The `unit_fusion` random-circuit
parity tests confirm this empirically at
@f$\lVert \psi_{\mathrm{fused}} - \psi_{\mathrm{unfused}} \rVert_2
\le 10^{-10}@f$ on 200-gate circuits at @f$n = 6@f$; the bound is
floating-point roundoff in the accumulated @f$2\times 2@f$ product,
not algorithmic error.

## 9. Clifford Tableau Completeness

The Gottesman-Knill theorem (Gottesman 1997, Aaronson-Gottesman
2004 §II) states that every circuit built from
@f$\{H, S, \mathrm{CNOT}\}@f$ and Pauli-basis measurements on the
initial state @f$|0\rangle^{\otimes n}@f$ can be simulated in
polynomial time on a classical computer.  The tableau update rules
(Aaronson-Gottesman 2004, Table II) are *exact*: after each
Clifford gate the tableau represents the correct stabilizer /
destabilizer generators of the true post-gate state.  Any
discrepancy between `unit_clifford` and a dense reference on a
Clifford circuit indicates a bug in the tableau update, not a
numerical approximation (this is why the tests demand exact
agreement, not a tolerance-bounded one).

## 10. Clifford-Assisted MPS (CA-MPS)

Hybrid representation
@f$\lvert\psi\rangle = D \lvert\phi\rangle@f$
with @f$D@f$ a Clifford unitary stored as an Aaronson-Gottesman
tableau and @f$\lvert\phi\rangle@f$ an MPS.  Operations split:

- **Clifford gate @f$U \in \mathcal{C}_n@f$**: update only the
  tableau, @f$D \to U D@f$.  Cost @f$O(n)@f$ for 1-qubit, @f$O(n)@f$
  for 2-qubit.  No MPS work.
- **Non-Clifford gate** parametrised as
  @f$\exp(-i\theta P/2)@f$ for some Pauli string @f$P@f$:
  conjugate through the tableau,
  @f$P' = D^\dagger P D \in \pm \mathcal{P}_n@f$, then apply the
  Pauli-rotation MPO @f$\exp(-i\theta P'/2)@f$ to
  @f$\lvert\phi\rangle@f$.

**Expectation value of a Pauli string @f$Q@f$.** The state
@f$\lvert\psi\rangle = D\lvert\phi\rangle@f$ has
@f[
  \langle\psi\rvert Q \lvert\psi\rangle
  = \langle\phi\rvert D^\dagger Q D \lvert\phi\rangle
  = \pm \langle\phi\rvert Q' \lvert\phi\rangle
@f]
where @f$Q' = D^\dagger Q D@f$ is computed by the tableau in
@f$O(n^2)@f$ and the inner expectation is the standard MPS Pauli
expectation.

**Entanglement bound.** Because Clifford unitaries on n qubits
realise an irreducible representation of the symplectic group
@f$\operatorname{Sp}(2n, \mathbb{F}_2)@f$, every CA-MPS factorisation
satisfies
@f[
  S(\lvert\psi\rangle) = S(D\lvert\phi\rangle)
                       \le S(\lvert\phi\rangle) + n_{\text{cut}}\log 2,
@f]
where @f$n_{\text{cut}}@f$ is the number of qubits the cut crosses
(2-local CNOTs in @f$D@f$ can introduce up to @f$n_{\text{cut}}\log 2@f$
of additional half-cut entropy when they straddle the bipartition).
For purely stabilizer @f$\lvert\psi\rangle@f$ a suitable @f$D@f$ takes
@f$\lvert\phi\rangle = \lvert 0^n\rangle@f$ and
@f$S(\lvert\phi\rangle) = 0@f$ exactly.

Full design and proofs: `docs/research/ca_mps.md`.

## 11. Variational-D (var-D) ground-state search

Given a Hamiltonian
@f$H = \sum_k c_k P_k@f$ as a Pauli sum, var-D minimises
@f[
  E(D, \lvert\phi\rangle)
  = \langle\psi\rvert H \lvert\psi\rangle
  = \sum_k c_k \langle\phi\rvert D^\dagger P_k D \lvert\phi\rangle
@f]
by alternating between a D-update (greedy local Clifford search) and
a @f$\lvert\phi\rangle@f$-update (imag-time Trotter sweep).

**D-update.** At fixed @f$\lvert\phi\rangle@f$, enumerate single-
and two-qubit Clifford candidates @f$g \in \{H, S, S^\dagger,
\mathrm{CNOT}, \mathrm{CZ}, \mathrm{SWAP}\}@f$.  For each candidate
compute @f$\Delta E = \langle\phi\rvert D^\dagger g^\dagger H g D \lvert\phi\rangle - \langle\phi\rvert D^\dagger H D\lvert\phi\rangle@f$.
Accept the largest @f$\Delta E < -\epsilon@f$.  Repeat to local
minimum.  Optional 2-gate composite moves @f$(g_1, g_2)@f$ help
escape 1-gate local minima where the right descent direction
requires two consecutive accepts.

**@f$\lvert\phi\rangle@f$-update.** First-order Trotter sweep over
the Pauli sum, each @f$\exp(-d\tau\, c_k\, P_k)@f$ applied via
the imag-time Pauli-rotation MPO with intermediate renormalisation
every 4 terms (without renormalisation, the MPS norm drifts and
the variational energy estimate falls below the true ground state).

**Convergence.** The alternating loop is monotone in
@f$E@f$ at each accepted gate and at each Trotter cycle, so
@f$E_t \to E_\infty@f$.  The fixed point need not be the global
ground state — the greedy gate search has stationary points
wherever no single (or 2-gate composite) Clifford reduces the
energy at fixed @f$\lvert\phi\rangle@f$.  Empirically TFIM converges
to within @f$10^{-3}@f$ of exact GS energy at n=12 with the
DUAL_TFIM warmstart; XXZ near criticality stalls at a
ferromagnetic-cluster local minimum (n=10) and needs FERRO_TFIM
warmstart or longer composite-2-gate search.

Implementation: `src/algorithms/tensor_network/ca_mps_var_d.{c,h}`,
public entry `moonlab_ca_mps_optimize_var_d_alternating`.

## 12. Gauge-aware warmstart via symplectic Gauss-Jordan

Given k pairwise-commuting Pauli generators
@f$\{g_0, \ldots, g_{k-1}\}@f$ on n qubits (the stabilizer subgroup
of an LGT, surface code, etc.), the Aaronson-Gottesman canonical-form
construction yields a Clifford @f$D_S@f$ such that
@f$D_S^\dagger g_i D_S = (-1)^{r_i} Z_{q_i}@f$ for distinct
qubits @f$q_i@f$ and phase bits @f$r_i \in \{0, 1\}@f$.  Then
@f[
  D_S \lvert b\rangle, \qquad b_{q_i} = r_i,
@f]
is in the simultaneous +1 eigenspace of every @f$g_i@f$.

**Algorithm.** Encode each generator as a row of an @f$F_2@f$
tableau @f$[X | Z | r]@f$ with shape @f$(k, 2n+1)@f$.  Symplectic
Gauss-Jordan with three operations:

1. **Row swap / row XOR**: re-order or multiply generators.  Free
   in gate cost.
2. **Single-qubit column ops** (H, S): swap or modify the @f$(x, z)@f$
   pair on one column.  H: @f$(x, z) \to (z, x)@f$, @f$r \xor= xz@f$.
   S: @f$(x, z) \to (x, x \xor z)@f$, @f$r \xor= xz@f$.
3. **Two-qubit column op** (CNOT @f$c \to t@f$):
   @f$x_t \xor= x_c@f$, @f$z_c \xor= z_t@f$,
   @f$r \xor= x_c \, z_t \, (x_t \oplus z_c \oplus 1)@f$.

The reduction loop pivots row-by-row on a fresh qubit, rotating
each non-trivial entry to X via H/S, clearing the row's other
non-trivial qubits via CNOT, eliminating the column from other
rows by row XOR, and finally rotating the pivot from X back to Z.
After @f$k@f$ pivots the tableau is in canonical form.

**Cost.** @f$O(k n^2)@f$ tableau ops, @f$O(n^2)@f$ Clifford gates
emitted.  For Z2 LGT with @f$k = N - 2 \sim n/2@f$ this is well
below the alternating-loop cost.

**Preparation circuit.**  The Cliffords @f$G_1, \ldots, G_M@f$
emitted as column ops satisfy
@f$g_i' = G_M \cdots G_1 g_i G_1^\dagger \cdots G_M^\dagger@f$.
The state stabilised by the original generators is
@f[
  \lvert\psi\rangle = G_1^\dagger G_2^\dagger \cdots G_M^\dagger \lvert b\rangle,
@f]
so the prep circuit applies X gates on the negative-phase pivots
(taking @f$\lvert 0^n\rangle@f$ to @f$\lvert b\rangle@f$) followed by
the recorded gate list in reverse with each @f$S@f$ replaced by
@f$S^\dagger@f$ and H, CNOT self-conjugate.

Implementation: `src/algorithms/tensor_network/ca_mps_var_d_stab_warmstart.{c,h}`,
public entry `moonlab_ca_mps_apply_stab_subgroup_warmstart`.  Math
write-up specific to 1+1D Z2 LGT in
`docs/research/var_d_lattice_gauge_theory.md`.

## References

All bounds and theorems cited above come from the references
enumerated in `ARCHITECTURE.md` §7. Primary citations for the
derivations in this document:

- Higham, *Functions of Matrices*, SIAM (2008): Newton-Schulz §5.
- Weisse, Wellein, Alvermann, Fehske, Rev. Mod. Phys. 78, 275 (2006):
  KPM error bounds.
- Bianco, Resta, Phys. Rev. B 84, 241106(R) (2011), arXiv:1111.5697:
  Eq. (2) and the factor @f$-4\pi/A_{\mathrm{uc}}@f$.
- Fukui, Hatsugai, Suzuki, JPSJ 74, 1674 (2005),
  arXiv:cond-mat/0503172: integer quantisation of the plaquette sum.
- Mezzadri, Notices AMS 54, 592 (2007), arXiv:math-ph/0609050:
  Haar-correct QR algorithm.
- Cross, Bishop, Sheldon, Nation, Gambetta, Phys. Rev. A 100, 032328
  (2019), arXiv:1811.12926: Quantum Volume heavy-output protocol.
- Cleve, Ekert, Macchiavello, Mosca, Proc. R. Soc. Lond. A 454, 339
  (1998), arXiv:quant-ph/9708016: QPE precision bound.
- Aaronson, Gottesman, Phys. Rev. A 70, 052328 (2004),
  arXiv:quant-ph/0406196: Clifford tableau update rules.
- J. K. Asbóth, L. Oroszlány and A. Pályi, *A Short Course on
  Topological Insulators: Band Structure and Edge States in One and
  Two Dimensions*, Lect. Notes Phys. 919, Springer (2016),
  doi:10.1007/978-3-319-25607-8. §1.5 for the SSH Zak-phase /
  winding-number sign convention adopted here.
