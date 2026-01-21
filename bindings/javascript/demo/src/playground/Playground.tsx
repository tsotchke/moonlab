import React, { useState, useCallback, useRef, useEffect } from 'react';
import { ensureMoonlabWorker, runCircuitInWorker, type WorkerGate } from '../workers/moonlabClient';
import './Playground.css';

// Gate categories for organized display
type SingleQubitGate = 'H' | 'X' | 'Y' | 'Z' | 'S' | 'T' | 'Sdg' | 'Tdg' | 'SX' | 'I';
type RotationGate = 'Rx' | 'Ry' | 'Rz' | 'P' | 'U';
type TwoQubitGate = 'CNOT' | 'CY' | 'CZ' | 'CH' | 'CP' | 'SWAP' | 'iSWAP' | 'DCX';
type ThreeQubitGate = 'CCX' | 'CCZ' | 'CSWAP';
type SpecialElement = 'M' | 'Reset' | 'Barrier';

type GateType = SingleQubitGate | RotationGate | TwoQubitGate | ThreeQubitGate | SpecialElement;

interface Gate {
  id: string;
  type: GateType;
  qubit: number;
  timeSlot: number;
  controlQubit?: number;
  controlQubit2?: number; // For 3-qubit gates
  targetQubit?: number;   // For multi-qubit gates
  angle?: number;         // For rotation gates
}

interface CircuitState {
  numQubits: number;
  numSlots: number;
  gates: Gate[];
}

interface GateInfo {
  name: string;
  description: string;
  color: string;
  category: 'single' | 'rotation' | 'two-qubit' | 'three-qubit' | 'special';
  symbol?: string;
  needsAngle?: boolean;
  needsControl?: boolean;
  needsTwoControls?: boolean;
}

const GATE_INFO: Record<GateType, GateInfo> = {
  // Single-qubit gates
  H: { name: 'Hadamard', description: 'Creates superposition |0⟩→|+⟩, |1⟩→|-⟩', color: '#6496ff', category: 'single' },
  X: { name: 'Pauli-X', description: 'Bit flip (NOT gate)', color: '#ff6666', category: 'single' },
  Y: { name: 'Pauli-Y', description: 'Bit + phase flip', color: '#66cc66', category: 'single' },
  Z: { name: 'Pauli-Z', description: 'Phase flip |1⟩→-|1⟩', color: '#ffcc44', category: 'single' },
  S: { name: 'S Gate', description: 'Phase gate π/2 (√Z)', color: '#ff66ff', category: 'single' },
  T: { name: 'T Gate', description: 'Phase gate π/4 (√S)', color: '#66ffff', category: 'single' },
  Sdg: { name: 'S†', description: 'S-dagger (inverse S)', color: '#cc44cc', category: 'single', symbol: 'S†' },
  Tdg: { name: 'T†', description: 'T-dagger (inverse T)', color: '#44cccc', category: 'single', symbol: 'T†' },
  SX: { name: '√X', description: 'Square root of X', color: '#ff8888', category: 'single', symbol: '√X' },
  I: { name: 'Identity', description: 'Identity (no operation)', color: '#888888', category: 'single' },

  // Rotation gates
  Rx: { name: 'Rx(θ)', description: 'X-axis rotation', color: '#ff4444', category: 'rotation', needsAngle: true },
  Ry: { name: 'Ry(θ)', description: 'Y-axis rotation', color: '#44ff44', category: 'rotation', needsAngle: true },
  Rz: { name: 'Rz(θ)', description: 'Z-axis rotation', color: '#ffff44', category: 'rotation', needsAngle: true },
  P: { name: 'Phase(θ)', description: 'Phase rotation', color: '#ff88ff', category: 'rotation', symbol: 'P', needsAngle: true },
  U: { name: 'U Gate', description: 'Universal single-qubit gate', color: '#8888ff', category: 'rotation', needsAngle: true },

  // Two-qubit gates
  CNOT: { name: 'CNOT', description: 'Controlled-X (CX)', color: '#9966ff', category: 'two-qubit', symbol: 'CX', needsControl: true },
  CY: { name: 'CY', description: 'Controlled-Y', color: '#66aa66', category: 'two-qubit', needsControl: true },
  CZ: { name: 'CZ', description: 'Controlled-Z', color: '#ff9966', category: 'two-qubit', needsControl: true },
  CH: { name: 'CH', description: 'Controlled-Hadamard', color: '#6699ff', category: 'two-qubit', needsControl: true },
  CP: { name: 'CPhase', description: 'Controlled-Phase', color: '#cc66ff', category: 'two-qubit', needsControl: true, needsAngle: true },
  SWAP: { name: 'SWAP', description: 'Swap two qubits', color: '#66ff99', category: 'two-qubit', symbol: '⨯' },
  iSWAP: { name: 'iSWAP', description: 'iSWAP gate', color: '#99ff66', category: 'two-qubit', symbol: 'i⨯' },
  DCX: { name: 'DCX', description: 'Double CNOT', color: '#aa66ff', category: 'two-qubit' },

  // Three-qubit gates
  CCX: { name: 'Toffoli', description: 'Controlled-Controlled-X (CCX)', color: '#ff66cc', category: 'three-qubit', symbol: 'CCX', needsTwoControls: true },
  CCZ: { name: 'CCZ', description: 'Controlled-Controlled-Z', color: '#ffaa66', category: 'three-qubit', needsTwoControls: true },
  CSWAP: { name: 'Fredkin', description: 'Controlled-SWAP', color: '#66ffcc', category: 'three-qubit', needsTwoControls: true },

  // Special elements
  M: { name: 'Measure', description: 'Measurement in Z basis', color: '#ffffff', category: 'special', symbol: 'M' },
  Reset: { name: 'Reset', description: 'Reset qubit to |0⟩', color: '#aaaaaa', category: 'special', symbol: '|0⟩' },
  Barrier: { name: 'Barrier', description: 'Visual separator (no operation)', color: '#555555', category: 'special', symbol: '│' },
};

const NUM_TIME_SLOTS = 12;
const MAX_QUBITS = 8;

// Gate categories for the palette
const GATE_CATEGORIES = {
  'Single Qubit': ['H', 'X', 'Y', 'Z', 'S', 'T', 'Sdg', 'Tdg', 'SX', 'I'] as GateType[],
  'Rotations': ['Rx', 'Ry', 'Rz', 'P', 'U'] as GateType[],
  'Two Qubit': ['CNOT', 'CY', 'CZ', 'CH', 'CP', 'SWAP', 'iSWAP', 'DCX'] as GateType[],
  'Three Qubit': ['CCX', 'CCZ', 'CSWAP'] as GateType[],
  'Special': ['M', 'Reset', 'Barrier'] as GateType[],
};

const Playground: React.FC = () => {
  const logoUrl = `${import.meta.env.BASE_URL}moonlab.png`;
  const [circuit, setCircuit] = useState<CircuitState>({
    numQubits: 3,
    numSlots: NUM_TIME_SLOTS,
    gates: [],
  });
  const [probabilities, setProbabilities] = useState<number[]>([]);
  const [selectedGate, setSelectedGate] = useState<GateType | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [expandedCategory, setExpandedCategory] = useState<string>('Single Qubit');
  const [controlMode, setControlMode] = useState<'target' | 'control1' | 'control2'>('target');
  const [pendingGate, setPendingGate] = useState<{ timeSlot: number; controlQubit?: number; controlQubit2?: number } | null>(null);
  const [rotationAngle, setRotationAngle] = useState<number>(Math.PI / 2);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const blochCanvasRef = useRef<HTMLCanvasElement>(null);

  const getGateAt = useCallback((qubit: number, timeSlot: number): Gate | undefined => {
    return circuit.gates.find(g =>
      (g.qubit === qubit || g.controlQubit === qubit || g.controlQubit2 === qubit || g.targetQubit === qubit)
      && g.timeSlot === timeSlot
    );
  }, [circuit.gates]);

  const isMultiQubitGate = (type: GateType): boolean => {
    const info = GATE_INFO[type];
    return info.needsControl || info.needsTwoControls || type === 'SWAP' || type === 'iSWAP' || type === 'DCX';
  };

  const addGate = useCallback((qubit: number, timeSlot: number) => {
    if (!selectedGate) return;

    const gateInfo = GATE_INFO[selectedGate];

    // Check if there's already a gate at this position
    const existingGate = getGateAt(qubit, timeSlot);
    if (existingGate && !pendingGate) {
      // Remove existing gate if clicking on it
      setCircuit(prev => ({
        ...prev,
        gates: prev.gates.filter(g => g.id !== existingGate.id),
      }));
      return;
    }

    // Handle multi-qubit gates
    if (isMultiQubitGate(selectedGate)) {
      if (gateInfo.needsTwoControls) {
        // Three-qubit gate: need two controls + target
        if (controlMode === 'control1') {
          setPendingGate({ timeSlot, controlQubit: qubit });
          setControlMode('control2');
          return;
        } else if (controlMode === 'control2' && pendingGate) {
          if (qubit === pendingGate.controlQubit) return; // Same qubit
          setPendingGate({ ...pendingGate, controlQubit2: qubit });
          setControlMode('target');
          return;
        } else if (controlMode === 'target' && pendingGate?.controlQubit !== undefined && pendingGate?.controlQubit2 !== undefined) {
          if (qubit === pendingGate.controlQubit || qubit === pendingGate.controlQubit2) return;
          const newGate: Gate = {
            id: `gate-${Date.now()}-${Math.random()}`,
            type: selectedGate,
            qubit,
            timeSlot: pendingGate.timeSlot,
            controlQubit: pendingGate.controlQubit,
            controlQubit2: pendingGate.controlQubit2,
            angle: gateInfo.needsAngle ? rotationAngle : undefined,
          };
          setCircuit(prev => ({ ...prev, gates: [...prev.gates, newGate] }));
          setPendingGate(null);
          setControlMode('control1');
          return;
        }
      } else if (gateInfo.needsControl || selectedGate === 'SWAP' || selectedGate === 'iSWAP' || selectedGate === 'DCX') {
        // Two-qubit gate: need control + target
        if (controlMode === 'control1') {
          setPendingGate({ timeSlot, controlQubit: qubit });
          setControlMode('target');
          return;
        } else if (controlMode === 'target' && pendingGate?.controlQubit !== undefined) {
          if (qubit === pendingGate.controlQubit) return; // Same qubit
          const newGate: Gate = {
            id: `gate-${Date.now()}-${Math.random()}`,
            type: selectedGate,
            qubit,
            timeSlot: pendingGate.timeSlot,
            controlQubit: pendingGate.controlQubit,
            angle: gateInfo.needsAngle ? rotationAngle : undefined,
          };
          setCircuit(prev => ({ ...prev, gates: [...prev.gates, newGate] }));
          setPendingGate(null);
          setControlMode('control1');
          return;
        }
      }
    }

    // Single-qubit gate or first click for multi-qubit
    if (isMultiQubitGate(selectedGate)) {
      setPendingGate({ timeSlot });
      setControlMode(gateInfo.needsTwoControls ? 'control1' : 'control1');
      addGate(qubit, timeSlot); // Recursive call to start the flow
      return;
    }

    const newGate: Gate = {
      id: `gate-${Date.now()}-${Math.random()}`,
      type: selectedGate,
      qubit,
      timeSlot,
      angle: gateInfo.needsAngle ? rotationAngle : undefined,
    };

    setCircuit(prev => ({
      ...prev,
      gates: [...prev.gates, newGate],
    }));
  }, [selectedGate, getGateAt, pendingGate, controlMode, rotationAngle]);

  const clearCircuit = useCallback(() => {
    setCircuit(prev => ({ ...prev, gates: [] }));
    setProbabilities([]);
    setPendingGate(null);
    setControlMode('control1');
  }, []);

  const setNumQubits = useCallback((n: number) => {
    setCircuit({ numQubits: n, numSlots: NUM_TIME_SLOTS, gates: [] });
    setProbabilities([]);
    setPendingGate(null);
  }, []);

  const selectGate = useCallback((gate: GateType) => {
    if (selectedGate === gate) {
      setSelectedGate(null);
      setPendingGate(null);
      setControlMode('control1');
    } else {
      setSelectedGate(gate);
      setPendingGate(null);
      setControlMode(GATE_INFO[gate].needsTwoControls ? 'control1' :
                     (GATE_INFO[gate].needsControl || gate === 'SWAP' || gate === 'iSWAP' || gate === 'DCX') ? 'control1' : 'target');
    }
  }, [selectedGate]);

  const simulateCircuit = useCallback(async () => {
    setIsSimulating(true);
    try {
      await ensureMoonlabWorker();
      const sortedGates = [...circuit.gates].sort((a, b) => a.timeSlot - b.timeSlot);
      const workerGates: WorkerGate[] = sortedGates.map((gate) => ({
        type: gate.type,
        qubit: gate.qubit,
        controlQubit: gate.controlQubit,
        controlQubit2: gate.controlQubit2,
        angle: gate.angle,
      }));
      const result = await runCircuitInWorker({
        numQubits: circuit.numQubits,
        gates: workerGates,
      });
      if (result.warnings.length) {
        console.warn('Playground gate warnings:', result.warnings);
      }
      setProbabilities(Array.from(result.probabilities));
    } catch (error) {
      console.error('Failed to simulate circuit', error);
      setProbabilities([]);
    } finally {
      setIsSimulating(false);
    }
  }, [circuit]);

  // Draw amplitude bars
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#0d0d1a';
    ctx.fillRect(0, 0, width, height);

    // If no probabilities, show placeholder
    if (probabilities.length === 0) {
      ctx.fillStyle = '#404060';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Click "Simulate" to see results', width / 2, height / 2);
      return;
    }

    const barWidth = Math.max(4, (width - 40) / probabilities.length - 2);
    const maxHeight = height - 60;

    // Draw bars
    probabilities.forEach((prob, i) => {
      const x = 20 + i * (barWidth + 2);
      const barHeight = prob * maxHeight;
      const y = height - 30 - barHeight;

      // Bar gradient
      const gradient = ctx.createLinearGradient(x, y + barHeight, x, y);
      gradient.addColorStop(0, '#6496ff');
      gradient.addColorStop(1, '#9966ff');
      ctx.fillStyle = gradient;
      ctx.fillRect(x, y, barWidth, barHeight);

      // State label
      if (probabilities.length <= 16) {
        ctx.fillStyle = '#606080';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        const label = i.toString(2).padStart(circuit.numQubits, '0');
        ctx.fillText(`|${label}⟩`, x + barWidth / 2, height - 10);
      }
    });

    // Y-axis
    ctx.strokeStyle = '#303050';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(15, 20);
    ctx.lineTo(15, height - 30);
    ctx.stroke();

    // Y-axis labels
    ctx.fillStyle = '#606080';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText('1.0', 12, 25);
    ctx.fillText('0.5', 12, height / 2);
    ctx.fillText('0.0', 12, height - 32);
  }, [probabilities, circuit.numQubits]);

  // Draw simple Bloch sphere representation
  useEffect(() => {
    const canvas = blochCanvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) / 2 - 30;

    ctx.fillStyle = '#0d0d1a';
    ctx.fillRect(0, 0, width, height);

    // Draw sphere outline
    ctx.strokeStyle = 'rgba(100, 150, 255, 0.3)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.stroke();

    // Draw axes
    const drawAxis = (angle: number, label: string, color: string) => {
      const x = cx + Math.cos(angle) * radius;
      const y = cy + Math.sin(angle) * radius;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(x, y);
      ctx.stroke();

      ctx.fillStyle = color;
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(label, x + Math.cos(angle) * 15, y + Math.sin(angle) * 15 + 4);
    };

    drawAxis(-Math.PI / 2, '|0⟩', '#00ff88');
    drawAxis(Math.PI / 2, '|1⟩', '#ff6688');
    drawAxis(0, '+X', '#6496ff');
    drawAxis(Math.PI, '-X', '#6496ff');

    // Draw state vector based on probabilities
    if (probabilities.length === 2) {
      const theta = Math.acos(Math.sqrt(probabilities[0])) * 2;
      const stateX = cx + Math.sin(theta) * radius * 0.9;
      const stateY = cy - Math.cos(theta) * radius * 0.9;

      // State vector arrow
      ctx.strokeStyle = '#ffcc00';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(stateX, stateY);
      ctx.stroke();

      // Arrow head
      ctx.fillStyle = '#ffcc00';
      ctx.beginPath();
      ctx.arc(stateX, stateY, 6, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [probabilities]);

  const getPlacementHint = (): string => {
    if (!selectedGate) return 'Select a gate from the palette';
    const info = GATE_INFO[selectedGate];
    if (info.needsTwoControls) {
      if (controlMode === 'control1') return `Click to place first control for ${info.name}`;
      if (controlMode === 'control2') return `Click to place second control for ${info.name}`;
      return `Click to place target for ${info.name}`;
    }
    if (info.needsControl || selectedGate === 'SWAP' || selectedGate === 'iSWAP' || selectedGate === 'DCX') {
      if (controlMode === 'control1') return `Click to place ${selectedGate === 'SWAP' || selectedGate === 'iSWAP' ? 'first qubit' : 'control'} for ${info.name}`;
      return `Click to place ${selectedGate === 'SWAP' || selectedGate === 'iSWAP' ? 'second qubit' : 'target'} for ${info.name}`;
    }
    return `Click to place ${info.name}`;
  };

  return (
    <div className="playground">
      <div className="section-header">
        <img className="section-logo" src={logoUrl} alt="" aria-hidden="true" />
        <div className="section-header-text">
          <h1 className="section-title">Quantum Circuit Playground</h1>
          <p className="section-description">
            Build quantum circuits interactively. Select a gate, then click on the circuit grid to place it.
          </p>
        </div>
      </div>

      <div className="playground-layout">
        {/* Gate Palette */}
        <div className="gate-palette">
          <h3 className="palette-title">Gates</h3>

          {Object.entries(GATE_CATEGORIES).map(([category, gates]) => (
            <div key={category} className="gate-category">
              <button
                className={`category-header ${expandedCategory === category ? 'expanded' : ''}`}
                onClick={() => setExpandedCategory(expandedCategory === category ? '' : category)}
              >
                <span>{category}</span>
                <span className="category-arrow">{expandedCategory === category ? '▼' : '▶'}</span>
              </button>
              {expandedCategory === category && (
                <div className="gate-grid">
                  {gates.map(gate => (
                    <button
                      key={gate}
                      className={`gate-btn ${selectedGate === gate ? 'selected' : ''}`}
                      style={{ borderColor: GATE_INFO[gate].color }}
                      onClick={() => selectGate(gate)}
                      title={GATE_INFO[gate].description}
                    >
                      <span className="gate-symbol" style={{ color: GATE_INFO[gate].color }}>
                        {GATE_INFO[gate].symbol || gate}
                      </span>
                      <span className="gate-name">{GATE_INFO[gate].name}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}

          {/* Rotation angle control */}
          {selectedGate && GATE_INFO[selectedGate].needsAngle && (
            <div className="palette-section">
              <h4>Rotation Angle</h4>
              <div className="angle-control">
                <input
                  type="range"
                  min="0"
                  max={2 * Math.PI}
                  step={Math.PI / 8}
                  value={rotationAngle}
                  onChange={(e) => setRotationAngle(parseFloat(e.target.value))}
                />
                <span className="angle-value">{(rotationAngle / Math.PI).toFixed(2)}π</span>
              </div>
              <div className="angle-presets">
                {[['π/4', Math.PI/4], ['π/2', Math.PI/2], ['π', Math.PI], ['3π/2', 3*Math.PI/2]].map(([label, val]) => (
                  <button
                    key={label}
                    className={`angle-preset ${Math.abs(rotationAngle - (val as number)) < 0.01 ? 'selected' : ''}`}
                    onClick={() => setRotationAngle(val as number)}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="palette-section">
            <h4>Qubits ({circuit.numQubits})</h4>
            <div className="qubit-selector">
              {Array.from({ length: MAX_QUBITS }, (_, i) => i + 1).map(n => (
                <button
                  key={n}
                  className={`qubit-btn ${circuit.numQubits === n ? 'selected' : ''}`}
                  onClick={() => setNumQubits(n)}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          <div className="palette-actions">
            <button className="btn btn-primary" onClick={simulateCircuit} disabled={isSimulating}>
              {isSimulating ? 'Simulating...' : 'Simulate'}
            </button>
            <button className="btn btn-secondary" onClick={clearCircuit}>
              Clear
            </button>
          </div>
        </div>

        {/* Circuit Editor */}
        <div className="circuit-editor">
          <h3>Circuit</h3>
          <div className="circuit-grid-container">
            {/* Time slot headers */}
            <div className="time-slot-header">
              <div className="qubit-label-spacer"></div>
              {Array.from({ length: circuit.numSlots }, (_, t) => (
                <div key={t} className="time-slot-label">t{t}</div>
              ))}
              <div className="measurement-spacer"></div>
            </div>

            {/* Qubit wires with grid cells */}
            {Array.from({ length: circuit.numQubits }, (_, q) => (
              <div key={q} className="qubit-row">
                <span className="qubit-label">q{q}</span>
                <div className="wire-grid">
                  {Array.from({ length: circuit.numSlots }, (_, t) => {
                    const gate = getGateAt(q, t);
                    const isPendingSlot = pendingGate?.timeSlot === t;
                    const isControl1 = pendingGate?.controlQubit === q && isPendingSlot;
                    const isControl2 = pendingGate?.controlQubit2 === q && isPendingSlot;

                    // Check if this cell is part of a multi-qubit gate
                    const multiGate = circuit.gates.find(g =>
                      g.timeSlot === t && (g.controlQubit === q || g.controlQubit2 === q)
                    );
                    const isControlNode = !!multiGate && (multiGate.controlQubit === q || multiGate.controlQubit2 === q);

                    // Render gate symbol
                    const renderGateSymbol = () => {
                      if (isControl1 || isControl2) {
                        return (
                          <div className="control-node pending" title="Control qubit (pending)">●</div>
                        );
                      }
                      if (isControlNode && multiGate) {
                        return (
                          <div
                            className="control-node"
                            style={{ color: GATE_INFO[multiGate.type].color }}
                            title={`Control for ${multiGate.type}`}
                          >●</div>
                        );
                      }
                      if (gate && gate.qubit === q) {
                        return (
                          <div
                            className={`gate-on-grid ${gate.type === 'Barrier' ? 'barrier' : ''}`}
                            style={{ backgroundColor: GATE_INFO[gate.type].color }}
                            title={`${GATE_INFO[gate.type].name}${gate.angle ? ` (${(gate.angle / Math.PI).toFixed(2)}π)` : ''}`}
                          >
                            {GATE_INFO[gate.type].symbol || gate.type}
                          </div>
                        );
                      }
                      return null;
                    };

                    return (
                      <div
                        key={t}
                        className={`grid-cell ${selectedGate ? 'clickable' : ''} ${gate ? 'has-gate' : ''} ${isPendingSlot ? 'pending-slot' : ''}`}
                        onClick={() => addGate(q, t)}
                      >
                        <div className="wire-segment"></div>
                        {renderGateSymbol()}
                        {/* Draw connection line for multi-qubit gates */}
                        {gate && gate.qubit === q && gate.controlQubit !== undefined && (
                          <div
                            className="gate-connection"
                            style={{
                              height: `${Math.abs(gate.controlQubit - q) * 50}px`,
                              top: gate.controlQubit < q ? `${(gate.controlQubit - q) * 50 + 25}px` : '25px',
                              borderColor: GATE_INFO[gate.type].color,
                            }}
                          />
                        )}
                      </div>
                    );
                  })}
                </div>
                <span className="measurement">M</span>
              </div>
            ))}
          </div>
          <p className={`hint ${pendingGate ? 'pending' : ''}`}>{getPlacementHint()}</p>
          {pendingGate && (
            <button className="btn btn-cancel" onClick={() => { setPendingGate(null); setControlMode('control1'); }}>
              Cancel Placement
            </button>
          )}
        </div>

        {/* Visualization */}
        <div className="visualization-panel">
          <div className="viz-card">
            <h3>Probability Amplitudes</h3>
            <canvas ref={canvasRef} width={400} height={200} className="viz-canvas" />
          </div>
          {circuit.numQubits === 1 && (
            <div className="viz-card">
              <h3>Bloch Sphere</h3>
              <canvas ref={blochCanvasRef} width={200} height={200} className="viz-canvas" />
            </div>
          )}
          {probabilities.length > 0 && (
            <div className="viz-card stats-card">
              <h3>Statistics</h3>
              <div className="stats-grid">
                <div className="stat">
                  <span className="stat-label">States</span>
                  <span className="stat-value">{probabilities.length}</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Non-zero</span>
                  <span className="stat-value">
                    {probabilities.filter(p => p > 0.001).length}
                  </span>
                </div>
                <div className="stat">
                  <span className="stat-label">Max Prob</span>
                  <span className="stat-value">
                    {(Math.max(...probabilities) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Playground;
