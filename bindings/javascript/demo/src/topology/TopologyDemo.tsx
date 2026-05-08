/**
 * Topology Explorer - interactive parameter sweeps for the QGT
 * topological-band-structure models shipped in moonlab v0.3.
 *
 * Each model has a closed-form analytical phase boundary, so we
 * compute the topological invariant directly in TS rather than
 * routing through the WASM bridge.  The numerical values match
 * what the C-side qgt_berry_grid / qgt_z2_invariant integrators
 * return; this page is the educational counterpart to the C
 * `bench_topology_phase_diagrams` benchmark.
 */

import React, { useMemo, useState } from 'react';
import './TopologyDemo.css';

type Model = 'QWZ' | 'Haldane' | 'KaneMele' | 'BHZ' | 'Kitaev' | 'SSH';

interface ModelMeta {
  id: Model;
  label: string;
  invariantName: 'Chern' | 'Z_2' | 'winding';
  description: string;
}

const MODELS: ModelMeta[] = [
  {
    id: 'QWZ',
    label: 'QWZ',
    invariantName: 'Chern',
    description: 'Qi-Wu-Zhang 2-band Chern insulator on a square lattice. Topological for -2 < m < 0 (C=+1) and 0 < m < 2 (C=-1).',
  },
  {
    id: 'Haldane',
    label: 'Haldane',
    invariantName: 'Chern',
    description: 'Honeycomb Chern insulator without uniform magnetic field. Topological for |M| < 3*sqrt(3)*|t2*sin(phi)|.',
  },
  {
    id: 'KaneMele',
    label: 'Kane-Mele',
    invariantName: 'Z_2',
    description: '4-band quantum spin Hall (QSH) on a honeycomb lattice. Z_2 = 1 for |lambda_v| < 3*sqrt(3)*|lambda_so|.',
  },
  {
    id: 'BHZ',
    label: 'BHZ',
    invariantName: 'Z_2',
    description: '4-band TI on a square lattice (HgTe quantum well). Lattice regularisation gives QSH for 0 < M/B < 8.',
  },
  {
    id: 'Kitaev',
    label: 'Kitaev p-wave',
    invariantName: 'Z_2',
    description: '1D BdG topological superconductor. Z_2 = 1 (Majorana edges) for |mu| < 2|t|.',
  },
  {
    id: 'SSH',
    label: 'SSH',
    invariantName: 'winding',
    description: 'Su-Schrieffer-Heeger 1D chiral model. Topological winding W = 1 when |t2| > |t1|.',
  },
];

interface ModelParams {
  /** QWZ */ m?: number;
  /** Haldane */ M?: number; t1?: number; t2?: number; phi?: number;
  /** Kane-Mele */ lambda_so?: number; lambda_v?: number;
  /** BHZ */ A?: number; B?: number; M_BHZ?: number;
  /** Kitaev */ t?: number; mu?: number;
  /** SSH */ ssh_t1?: number; ssh_t2?: number;
}

const DEFAULT_PARAMS: Record<Model, ModelParams> = {
  QWZ:      { m: -1.5 },
  Haldane:  { t1: 1.0, t2: 0.06, phi: Math.PI / 2, M: 0.1 },
  KaneMele: { lambda_so: 0.06, lambda_v: 0.1 },
  BHZ:      { A: 1.0, B: 1.0, M_BHZ: 2.0 },
  Kitaev:   { t: 1.0, mu: 0.0 },
  SSH:      { ssh_t1: 1.0, ssh_t2: 1.5 },
};

interface PhasePoint {
  param: number;
  invariant: number;
}

/**
 * Compute the topological invariant for a given model at a given
 * primary parameter value.  All models have closed-form analytical
 * phase boundaries -- this matches what the C-side QGT integrators
 * compute numerically.
 */
function invariantAt(model: Model, params: ModelParams, primary: number): number {
  switch (model) {
    case 'QWZ': {
      const m = primary;
      if (m > -2.0 && m < 0.0) return +1;
      if (m > 0.0 && m < 2.0) return -1;
      return 0;
    }
    case 'Haldane': {
      const M = primary;
      const t2 = params.t2 ?? 0.06;
      const phi = params.phi ?? Math.PI / 2;
      const boundary = 3.0 * Math.sqrt(3.0) * Math.abs(t2 * Math.sin(phi));
      return Math.abs(M) < boundary ? -1 : 0;
    }
    case 'KaneMele': {
      const lv = primary;
      const lso = params.lambda_so ?? 0.06;
      const boundary = 3.0 * Math.sqrt(3.0) * Math.abs(lso);
      return Math.abs(lv) < boundary ? 1 : 0;
    }
    case 'BHZ': {
      const M = primary;
      const B = params.B ?? 1.0;
      return (M / B > 0.0 && M / B < 8.0) ? 1 : 0;
    }
    case 'Kitaev': {
      const mu = primary;
      const t = params.t ?? 1.0;
      return Math.abs(mu) < 2.0 * Math.abs(t) ? 1 : 0;
    }
    case 'SSH': {
      const t2 = primary;
      const t1 = params.ssh_t1 ?? 1.0;
      return Math.abs(t2) > Math.abs(t1) ? 1 : 0;
    }
  }
}

interface PrimaryAxis {
  name: string;
  label: string;
  min: number;
  max: number;
  step: number;
  paramKey: keyof ModelParams;
  setKey: keyof ModelParams;
}

const PRIMARY_AXIS: Record<Model, PrimaryAxis> = {
  QWZ:      { name: 'm',         label: 'mass m',           min: -3.0, max: 3.0, step: 0.05, paramKey: 'm',         setKey: 'm' },
  Haldane:  { name: 'M',         label: 'sublattice mass M', min: -1.0, max: 1.0, step: 0.02, paramKey: 'M',         setKey: 'M' },
  KaneMele: { name: 'lambda_v',  label: 'sublattice mass λ_v', min:  0.0, max: 0.6, step: 0.01, paramKey: 'lambda_v', setKey: 'lambda_v' },
  BHZ:      { name: 'M',         label: 'mass M',            min: -2.0, max: 10.0, step: 0.1, paramKey: 'M_BHZ',     setKey: 'M_BHZ' },
  Kitaev:   { name: 'mu',        label: 'chemical potential μ', min: -3.0, max: 3.0, step: 0.05, paramKey: 'mu',     setKey: 'mu' },
  SSH:      { name: 't2',        label: 'inter-cell hopping t₂', min:  0.0, max: 2.0, step: 0.02, paramKey: 'ssh_t2', setKey: 'ssh_t2' },
};

function colorForInvariant(c: number): string {
  if (c === +1) return '#5cc8a8';   // teal-green: positive
  if (c === -1) return '#e07d5a';   // amber-orange: negative
  if (c === 0)  return '#666b75';   // muted grey: trivial
  return '#9b59b6';                  // purple: |C| >= 2 (Hofstadter etc.)
}

function symbolForInvariant(c: number, kind: 'Chern' | 'Z_2' | 'winding'): string {
  if (kind === 'Chern')   return c > 0 ? `+${c}` : `${c}`;
  if (kind === 'Z_2')     return c.toString();
  return c.toString();
}

const TopologyDemo: React.FC = () => {
  const [model, setModel] = useState<Model>('QWZ');
  const [params, setParams] = useState<ModelParams>(DEFAULT_PARAMS.QWZ);

  const meta = MODELS.find(m => m.id === model)!;
  const axis = PRIMARY_AXIS[model];

  const primaryValue = (params[axis.paramKey] as number | undefined) ?? 0;

  const handleModelChange = (newModel: Model) => {
    setModel(newModel);
    setParams(DEFAULT_PARAMS[newModel]);
  };

  const handlePrimaryChange = (v: number) => {
    setParams(p => ({ ...p, [axis.setKey]: v }));
  };

  const currentInvariant = useMemo(() =>
    invariantAt(model, params, primaryValue),
    [model, params, primaryValue]);

  // Phase ribbon: 60-step sweep across the primary axis.
  const phaseSweep: PhasePoint[] = useMemo(() => {
    const steps = 60;
    const results: PhasePoint[] = [];
    for (let i = 0; i <= steps; i++) {
      const p = axis.min + (axis.max - axis.min) * (i / steps);
      results.push({ param: p, invariant: invariantAt(model, params, p) });
    }
    return results;
  }, [model, params, axis]);

  // Locate phase boundaries (where invariant changes between consecutive points).
  const phaseBoundaries: number[] = useMemo(() => {
    const boundaries: number[] = [];
    for (let i = 1; i < phaseSweep.length; i++) {
      if (phaseSweep[i].invariant !== phaseSweep[i-1].invariant) {
        // Linear interpolation between the two points (boundary lies between).
        const p0 = phaseSweep[i-1].param;
        const p1 = phaseSweep[i].param;
        boundaries.push(0.5 * (p0 + p1));
      }
    }
    return boundaries;
  }, [phaseSweep]);

  return (
    <div className="topology-demo">
      <div className="topology-header">
        <h1>Topology Explorer</h1>
        <p className="topology-subtitle">
          Interactive phase diagrams for moonlab&apos;s n-band quantum-geometric-tensor
          module.  Slide the parameter to watch the topological invariant change.
        </p>
      </div>

      <div className="topology-model-tabs">
        {MODELS.map(m => (
          <button
            key={m.id}
            className={`topology-model-tab ${m.id === model ? 'active' : ''}`}
            onClick={() => handleModelChange(m.id)}
          >
            {m.label}
          </button>
        ))}
      </div>

      <div className="topology-layout">
        <aside className="topology-sidebar">
          <div className="topology-card">
            <h3>{meta.label}</h3>
            <p className="topology-description">{meta.description}</p>
          </div>

          <div className="topology-card">
            <h4>Parameters</h4>
            <div className="topology-slider-group">
              <label className="topology-slider-label">
                {axis.label}: <strong>{primaryValue.toFixed(3)}</strong>
              </label>
              <input
                type="range"
                min={axis.min}
                max={axis.max}
                step={axis.step}
                value={primaryValue}
                onChange={e => handlePrimaryChange(parseFloat(e.target.value))}
              />
              <div className="topology-axis-bounds">
                <span>{axis.min.toFixed(2)}</span>
                <span>{axis.max.toFixed(2)}</span>
              </div>
            </div>

            {/* Secondary parameters (read-only at v0.3 launch; full
                multi-knob interactivity lands in v0.3.x). */}
            {model === 'Haldane' && (
              <ReadOnlyParam label="t₁ (NN hopping)" value={params.t1?.toFixed(2)} />
            )}
            {model === 'Haldane' && (
              <ReadOnlyParam label="t₂ (NNN hopping)" value={params.t2?.toFixed(3)} />
            )}
            {model === 'Haldane' && (
              <ReadOnlyParam label="φ (Peierls phase)" value="π/2" />
            )}
            {model === 'KaneMele' && (
              <ReadOnlyParam label="λ_so (intrinsic SOC)" value={params.lambda_so?.toFixed(3)} />
            )}
            {model === 'BHZ' && (
              <ReadOnlyParam label="A (s-p hybridisation)" value={params.A?.toFixed(2)} />
            )}
            {model === 'BHZ' && (
              <ReadOnlyParam label="B (sub-band coupling)" value={params.B?.toFixed(2)} />
            )}
            {model === 'Kitaev' && (
              <ReadOnlyParam label="t (NN hopping)" value={params.t?.toFixed(2)} />
            )}
            {model === 'Kitaev' && (
              <ReadOnlyParam label="Δ (p-wave pairing)" value="0.5 (fixed)" />
            )}
            {model === 'SSH' && (
              <ReadOnlyParam label="t₁ (intra-cell)" value={params.ssh_t1?.toFixed(2)} />
            )}
          </div>
        </aside>

        <main className="topology-main">
          <div className="topology-invariant-display">
            <div className="topology-invariant-label">{meta.invariantName}</div>
            <div
              className="topology-invariant-value"
              style={{ color: colorForInvariant(currentInvariant) }}
            >
              {symbolForInvariant(currentInvariant, meta.invariantName)}
            </div>
            <div className="topology-invariant-status">
              {currentInvariant !== 0 ? 'topological' : 'trivial'}
            </div>
          </div>

          <div className="topology-ribbon-card">
            <h4>Phase ribbon</h4>
            <p className="topology-ribbon-desc">
              60-step sweep of {axis.label}.  Colour ↔ invariant value;
              vertical line marks current parameter.
            </p>
            <div className="topology-ribbon">
              {phaseSweep.map((pt, i) => {
                const isCurrent = Math.abs(pt.param - primaryValue) <
                  (axis.max - axis.min) / 120;
                return (
                  <div
                    key={i}
                    className={`topology-ribbon-cell ${isCurrent ? 'current' : ''}`}
                    style={{ backgroundColor: colorForInvariant(pt.invariant) }}
                    title={`${axis.name}=${pt.param.toFixed(3)} → ${meta.invariantName}=${pt.invariant}`}
                  />
                );
              })}
            </div>
            <div className="topology-ribbon-axis">
              <span>{axis.min.toFixed(1)}</span>
              <span className="topology-ribbon-axis-name">{axis.label}</span>
              <span>{axis.max.toFixed(1)}</span>
            </div>
            {phaseBoundaries.length > 0 && (
              <div className="topology-boundary-list">
                <strong>Phase boundaries:</strong>{' '}
                {phaseBoundaries.map((b, i) => (
                  <span key={i} className="topology-boundary-tag">
                    {axis.name} ≈ {b.toFixed(3)}
                  </span>
                ))}
              </div>
            )}
          </div>

          <div className="topology-legend-card">
            <h4>Legend</h4>
            <div className="topology-legend-row">
              <LegendItem color={colorForInvariant(+1)} label={`${meta.invariantName} = +1`} />
              <LegendItem color={colorForInvariant(-1)} label={`${meta.invariantName} = -1`} />
              <LegendItem color={colorForInvariant(0)} label={`${meta.invariantName} = 0 (trivial)`} />
            </div>
          </div>

          <div className="topology-physics-note">
            <h4>Physics note</h4>
            <PhysicsNote model={model} />
          </div>
        </main>
      </div>
    </div>
  );
};

const ReadOnlyParam: React.FC<{ label: string; value?: string }> = ({ label, value }) => (
  <div className="topology-readonly-param">
    <span className="topology-readonly-label">{label}</span>
    <span className="topology-readonly-value">{value}</span>
  </div>
);

const LegendItem: React.FC<{ color: string; label: string }> = ({ color, label }) => (
  <div className="topology-legend-item">
    <div className="topology-legend-swatch" style={{ backgroundColor: color }} />
    <span>{label}</span>
  </div>
);

const PhysicsNote: React.FC<{ model: Model }> = ({ model }) => {
  switch (model) {
    case 'QWZ':
      return (
        <>
          <p>
            The Qi-Wu-Zhang (QWZ) Hamiltonian (Qi, Wu, and Zhang,
            <em> Phys. Rev. B</em> <strong>74</strong>, 085308, 2006)
            exhibits gap closings at <code>m = -2, 0, +2</code>; the
            integer Chern number jumps by ±1 across each closing.  In
            the topological window <code>0 &lt; m &lt; 2</code> the
            Berry curvature is concentrated near <code>k = (π, π)</code>.
          </p>
          <p>
            Moonlab computes the Chern number through three independent
            paths: the Fukui-Hatsugai-Suzuki link-variable integrator
            <code>qgt_berry_grid</code>, the gauge-free projector trace
            <code>qgt_berry_grid_proj</code>, and the real-space
            Bianco-Resta marker <code>chern_marker</code>.  All three
            return identical integers on every gapped phase point.
          </p>
        </>
      );
    case 'Haldane':
      return (
        <p>
          The Haldane model (Haldane, <em>Phys. Rev. Lett.</em>{' '}
          <strong>61</strong>, 2015, 1988) was the first demonstration
          that a Chern insulator can arise without a uniform magnetic
          field.  Complex next-nearest-neighbour hopping with Peierls
          phase <code>φ</code> generates a staggered local flux that
          gaps the Dirac points at <code>K</code> and <code>K&apos;</code>{' '}
          with opposite signs, while preserving zero net flux per unit
          cell.  The phase boundary is <code>|M| = 3√3 |t₂ sin(φ)|</code>.
        </p>
      );
    case 'KaneMele':
      return (
        <p>
          The Kane-Mele model (Kane and Mele, <em>Phys. Rev. Lett.</em>{' '}
          <strong>95</strong>, 146802, 2005) is the canonical
          two-dimensional quantum spin Hall (QSH) insulator: a four-band,
          time-reversal-symmetric Hamiltonian classified by a Z₂
          invariant rather than an integer Chern number.  In the
          S<sub>z</sub>-conserving regime (Rashba coupling absent) the
          Z₂ invariant reduces to the parity of the spin-up sub-block
          Chern number.  The topological phase hosts helical edge modes
          carrying counter-propagating spin currents.
        </p>
      );
    case 'BHZ':
      return (
        <p>
          The Bernevig-Hughes-Zhang model (Bernevig, Hughes, and Zhang,{' '}
          <em>Science</em> <strong>314</strong>, 1757, 2006) models HgTe
          quantum wells.  Its lattice regularisation has gap closings at
          {' '}<code>M = 0</code> (Γ point), <code>M = 4B</code>{' '}
          (X-corners, which cancel pairwise), and <code>M = 8B</code>{' '}
          (M-corner).  The QSH phase occupies <code>0 &lt; M/B &lt; 8</code>;
          the symmetry-cancelled X-corner closings preserve the
          topology throughout this interval.
        </p>
      );
    case 'Kitaev':
      return (
        <p>
          The Kitaev p-wave chain (Kitaev, <em>Physics-Uspekhi</em>{' '}
          <strong>44</strong>, 131, 2001) is a one-dimensional topological
          superconductor in the BdG framework.  The Z₂ invariant — the
          product of Pfaffian signs at the time-reversal-invariant
          momenta <code>k = 0, π</code> — equals 1 when{' '}
          <code>|μ| &lt; 2|t|</code>, and the corresponding open chain
          hosts Majorana zero modes at its boundaries.
        </p>
      );
    case 'SSH':
      return (
        <p>
          The Su-Schrieffer-Heeger model (Su, Schrieffer, and Heeger,{' '}
          <em>Phys. Rev. Lett.</em> <strong>42</strong>, 1698, 1979) is a
          one-dimensional bipartite chain with intra-cell hopping{' '}
          <code>t₁</code> and inter-cell hopping <code>t₂</code>.  Chiral
          symmetry quantises the winding number; the topological phase{' '}
          <code>W = 1</code> arises for <code>|t₂| &gt; |t₁|</code> and
          supports protected edge zero modes under open boundary
          conditions.
        </p>
      );
  }
};

export default TopologyDemo;
