/**
 * Circuit Diagram Visualization (Canvas 2D)
 *
 * Renders a quantum circuit as a visual diagram with gates,
 * qubit lines, and optional annotations.
 */

import type { Circuit, Gate } from '@moonlab/quantum-core';
import type {
  BaseVisualizationOptions,
  ColorSchemeOptions,
  Point2D,
  EventListener,
  InteractionEvent,
} from '../types';
import { clamp } from '../types';

// ============================================================================
// Types
// ============================================================================

export interface CircuitDiagramOptions
  extends BaseVisualizationOptions,
    ColorSchemeOptions {
  /**
   * Cell width for each gate slot (default: 60)
   */
  cellWidth?: number;

  /**
   * Cell height for each qubit line (default: 50)
   */
  cellHeight?: number;

  /**
   * Show qubit labels (default: true)
   */
  showLabels?: boolean;

  /**
   * Show gate names (default: true)
   */
  showGateNames?: boolean;

  /**
   * Gate style (default: 'box')
   */
  gateStyle?: 'box' | 'circle' | 'ibm';

  /**
   * Line width for qubit wires (default: 2)
   */
  wireWidth?: number;

  /**
   * Padding around diagram (default: { top: 20, right: 20, bottom: 20, left: 60 })
   */
  padding?: { top: number; right: number; bottom: number; left: number };

  /**
   * Gate colors by type
   */
  gateColors?: Record<string, string>;

  /**
   * Enable horizontal scrolling for large circuits (default: true)
   */
  scrollable?: boolean;
}

interface GatePosition {
  gate: Gate;
  column: number;
  qubits: number[];
}

// ============================================================================
// Default Gate Colors
// ============================================================================

const DEFAULT_GATE_COLORS: Record<string, string> = {
  // Single-qubit gates
  h: '#4f46e5', // Indigo
  x: '#dc2626', // Red
  y: '#16a34a', // Green
  z: '#2563eb', // Blue
  s: '#7c3aed', // Purple
  sdg: '#7c3aed',
  t: '#db2777', // Pink
  tdg: '#db2777',
  rx: '#ea580c', // Orange
  ry: '#ea580c',
  rz: '#ea580c',
  phase: '#0891b2', // Cyan
  u3: '#6366f1', // Indigo

  // Two-qubit gates
  cnot: '#059669', // Emerald
  cx: '#059669',
  cz: '#0284c7', // Sky
  cy: '#16a34a',
  swap: '#8b5cf6', // Violet
  cphase: '#0891b2',
  crx: '#f97316',
  cry: '#f97316',
  crz: '#f97316',

  // Three-qubit gates
  toffoli: '#b45309', // Amber
  ccx: '#b45309',
  fredkin: '#9333ea', // Purple
  cswap: '#9333ea',

  // Multi-qubit
  qft: '#0d9488', // Teal
  iqft: '#0d9488',

  // Special
  measure: '#374151', // Gray
  barrier: '#9ca3af',
};

// ============================================================================
// CircuitDiagram Class
// ============================================================================

export class CircuitDiagram {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private options: Required<CircuitDiagramOptions>;
  private circuit: Circuit | null = null;
  private gatePositions: GatePosition[] = [];
  private hoveredGate: number = -1;
  private scrollX: number = 0;
  private listeners: Map<string, Set<EventListener>> = new Map();

  constructor(
    canvas: HTMLCanvasElement | string,
    options: CircuitDiagramOptions = {}
  ) {
    // Get canvas element
    if (typeof canvas === 'string') {
      const el = document.getElementById(canvas);
      if (!el || !(el instanceof HTMLCanvasElement)) {
        throw new Error(`Canvas element not found: ${canvas}`);
      }
      this.canvas = el;
    } else {
      this.canvas = canvas;
    }

    // Get 2D context
    const ctx = this.canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Failed to get 2D canvas context');
    }
    this.ctx = ctx;

    // Set default options
    this.options = {
      width: options.width ?? 800,
      height: options.height ?? 400,
      backgroundColor: options.backgroundColor ?? '#ffffff',
      animated: options.animated ?? true,
      animationDuration: options.animationDuration ?? 300,
      pixelRatio: options.pixelRatio ?? (typeof window !== 'undefined' ? window.devicePixelRatio : 1),
      scheme: options.scheme ?? 'default',
      primaryColor: options.primaryColor ?? '#4f46e5',
      secondaryColor: options.secondaryColor ?? '#06b6d4',
      textColor: options.textColor ?? '#1f2937',
      gridColor: options.gridColor ?? '#e5e7eb',
      cellWidth: options.cellWidth ?? 60,
      cellHeight: options.cellHeight ?? 50,
      showLabels: options.showLabels ?? true,
      showGateNames: options.showGateNames ?? true,
      gateStyle: options.gateStyle ?? 'box',
      wireWidth: options.wireWidth ?? 2,
      padding: options.padding ?? { top: 20, right: 20, bottom: 20, left: 60 },
      gateColors: { ...DEFAULT_GATE_COLORS, ...options.gateColors },
      scrollable: options.scrollable ?? true,
    };

    // Setup canvas
    this.setupCanvas();

    // Setup interaction
    this.setupInteraction();

    // Initial render
    this.render();
  }

  // =========================================================================
  // Public Methods
  // =========================================================================

  /**
   * Set circuit to display
   */
  setCircuit(circuit: Circuit): void {
    this.circuit = circuit;
    this.calculateLayout();
    this.scrollX = 0;
    this.render();
  }

  /**
   * Update options
   */
  setOptions(options: Partial<CircuitDiagramOptions>): void {
    Object.assign(this.options, options);
    if (options.gateColors) {
      this.options.gateColors = { ...DEFAULT_GATE_COLORS, ...options.gateColors };
    }
    if (options.width || options.height || options.pixelRatio) {
      this.setupCanvas();
    }
    if (this.circuit) {
      this.calculateLayout();
    }
    this.render();
  }

  /**
   * Scroll to a specific column
   */
  scrollTo(column: number): void {
    const maxScroll = this.getMaxScroll();
    this.scrollX = clamp(column * this.options.cellWidth, 0, maxScroll);
    this.render();
  }

  /**
   * Add event listener
   */
  on(event: string, callback: EventListener): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }

  /**
   * Remove event listener
   */
  off(event: string, callback: EventListener): void {
    this.listeners.get(event)?.delete(callback);
  }

  /**
   * Dispose and cleanup
   */
  dispose(): void {
    this.canvas.removeEventListener('mousemove', this.onMouseMove);
    this.canvas.removeEventListener('click', this.onClick);
    this.canvas.removeEventListener('wheel', this.onWheel);
    this.listeners.clear();
  }

  // =========================================================================
  // Private Methods
  // =========================================================================

  private setupCanvas(): void {
    const { width, height, pixelRatio } = this.options;

    this.canvas.width = width * pixelRatio;
    this.canvas.height = height * pixelRatio;
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;

    this.ctx.scale(pixelRatio, pixelRatio);
  }

  private setupInteraction(): void {
    this.onMouseMove = this.onMouseMove.bind(this);
    this.onClick = this.onClick.bind(this);
    this.onWheel = this.onWheel.bind(this);

    this.canvas.addEventListener('mousemove', this.onMouseMove);
    this.canvas.addEventListener('click', this.onClick);
    this.canvas.addEventListener('wheel', this.onWheel, { passive: false });
  }

  private getMaxScroll(): number {
    if (!this.circuit) return 0;
    const numColumns = Math.max(...this.gatePositions.map((g) => g.column)) + 1;
    const contentWidth = numColumns * this.options.cellWidth;
    const viewWidth = this.options.width - this.options.padding.left - this.options.padding.right;
    return Math.max(0, contentWidth - viewWidth);
  }

  private getGateAtPosition(x: number, y: number): number {
    if (!this.circuit) return -1;

    const { padding, cellWidth, cellHeight } = this.options;
    const contentX = x - padding.left + this.scrollX;
    const contentY = y - padding.top;

    for (let i = 0; i < this.gatePositions.length; i++) {
      const pos = this.gatePositions[i];
      const gateX = pos.column * cellWidth + cellWidth / 2;
      const gateY = (Math.min(...pos.qubits) + Math.max(...pos.qubits)) / 2 * cellHeight + cellHeight / 2;

      const dx = Math.abs(contentX - gateX);
      const dy = Math.abs(contentY - gateY);

      if (dx < cellWidth / 2 && dy < cellHeight / 2) {
        return i;
      }
    }

    return -1;
  }

  private onMouseMove(e: MouseEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const gateIndex = this.getGateAtPosition(x, y);

    if (gateIndex !== this.hoveredGate) {
      this.hoveredGate = gateIndex;
      this.canvas.style.cursor = gateIndex >= 0 ? 'pointer' : 'default';
      this.render();

      if (gateIndex >= 0) {
        const event: InteractionEvent = {
          type: 'hover',
          point: { x, y },
        };
        this.emit('hover', event);
      }
    }
  }

  private onClick(e: MouseEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const gateIndex = this.getGateAtPosition(x, y);

    if (gateIndex >= 0) {
      const event: InteractionEvent = {
        type: 'click',
        point: { x, y },
      };
      this.emit('click', event);
    }
  }

  private onWheel(e: WheelEvent): void {
    if (!this.options.scrollable) return;

    e.preventDefault();
    const maxScroll = this.getMaxScroll();
    this.scrollX = clamp(this.scrollX + e.deltaX + e.deltaY, 0, maxScroll);
    this.render();
  }

  private emit(event: string, data: unknown): void {
    this.listeners.get(event)?.forEach((cb) => cb(data));
  }

  private calculateLayout(): void {
    if (!this.circuit) {
      this.gatePositions = [];
      return;
    }

    this.gatePositions = [];
    const qubitColumns: number[] = new Array(this.circuit.numQubits).fill(0);

    for (const gate of this.circuit.gates) {
      const qubits = this.getGateQubits(gate);
      const minColumn = Math.max(...qubits.map((q) => qubitColumns[q]));

      this.gatePositions.push({
        gate,
        column: minColumn,
        qubits,
      });

      // Update qubit columns
      for (const q of qubits) {
        qubitColumns[q] = minColumn + 1;
      }
    }
  }

  private getGateQubits(gate: Gate): number[] {
    if ('qubit' in gate && !('control' in gate)) {
      return [gate.qubit];
    }
    if ('control' in gate && 'target' in gate) {
      return [gate.control, gate.target];
    }
    if ('qubits' in gate) {
      return [...gate.qubits];
    }
    return [];
  }

  private render(): void {
    const { ctx, options } = this;
    const { width, height } = options;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Background
    if (options.backgroundColor !== 'transparent') {
      ctx.fillStyle = options.backgroundColor as string;
      ctx.fillRect(0, 0, width, height);
    }

    if (!this.circuit) {
      this.drawEmptyState();
      return;
    }

    // Set clipping region for scrollable content
    ctx.save();
    ctx.beginPath();
    ctx.rect(
      options.padding.left,
      options.padding.top,
      width - options.padding.left - options.padding.right,
      height - options.padding.top - options.padding.bottom
    );
    ctx.clip();

    // Translate for scrolling
    ctx.translate(-this.scrollX, 0);

    // Draw qubit lines
    this.drawQubitLines();

    // Draw gates
    this.drawGates();

    ctx.restore();

    // Draw labels (not affected by scroll)
    if (options.showLabels) {
      this.drawLabels();
    }

    // Draw scroll indicators
    if (options.scrollable && this.getMaxScroll() > 0) {
      this.drawScrollIndicators();
    }
  }

  private drawEmptyState(): void {
    const { ctx, options } = this;

    ctx.save();
    ctx.font = '14px sans-serif';
    ctx.fillStyle = options.textColor as string;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('No circuit loaded', options.width / 2, options.height / 2);
    ctx.restore();
  }

  private drawQubitLines(): void {
    if (!this.circuit) return;

    const { ctx, options } = this;
    const { padding, cellHeight, wireWidth } = options;
    const numColumns = Math.max(...this.gatePositions.map((g) => g.column), 0) + 2;

    ctx.save();
    ctx.strokeStyle = options.textColor as string;
    ctx.lineWidth = wireWidth;

    for (let q = 0; q < this.circuit.numQubits; q++) {
      const y = padding.top + q * cellHeight + cellHeight / 2;

      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + numColumns * options.cellWidth, y);
      ctx.stroke();
    }

    ctx.restore();
  }

  private drawGates(): void {
    const { ctx, options } = this;
    const { padding, cellWidth, cellHeight, gateStyle, showGateNames } = options;

    for (let i = 0; i < this.gatePositions.length; i++) {
      const pos = this.gatePositions[i];
      const gate = pos.gate;
      const isHovered = i === this.hoveredGate;

      const centerX = padding.left + pos.column * cellWidth + cellWidth / 2;
      const qubits = pos.qubits;
      const color = options.gateColors[gate.type] || options.primaryColor;

      ctx.save();

      // Draw multi-qubit connectors
      if (qubits.length > 1) {
        const minQ = Math.min(...qubits);
        const maxQ = Math.max(...qubits);

        ctx.strokeStyle = color as string;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(centerX, padding.top + minQ * cellHeight + cellHeight / 2);
        ctx.lineTo(centerX, padding.top + maxQ * cellHeight + cellHeight / 2);
        ctx.stroke();
      }

      // Draw gate symbols
      for (const q of qubits) {
        const centerY = padding.top + q * cellHeight + cellHeight / 2;
        this.drawGateSymbol(gate, centerX, centerY, color as string, isHovered, gateStyle);
      }

      // Draw gate name
      if (showGateNames && gate.type !== 'barrier' && gate.type !== 'measure') {
        const centerY = padding.top + qubits[0] * cellHeight + cellHeight / 2;
        ctx.font = 'bold 12px sans-serif';
        ctx.fillStyle = '#fff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(this.getGateLabel(gate), centerX, centerY);
      }

      ctx.restore();
    }
  }

  private drawGateSymbol(
    gate: Gate,
    x: number,
    y: number,
    color: string,
    isHovered: boolean,
    style: 'box' | 'circle' | 'ibm'
  ): void {
    const { ctx, options } = this;
    const size = options.cellHeight * 0.6;

    ctx.save();

    if (isHovered) {
      ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';
      ctx.shadowBlur = 8;
      ctx.shadowOffsetY = 2;
    }

    // Handle special gates
    if (gate.type === 'barrier') {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, y - size / 2);
      ctx.lineTo(x, y + size / 2);
      ctx.stroke();
      ctx.restore();
      return;
    }

    if (gate.type === 'measure') {
      ctx.fillStyle = color;
      ctx.fillRect(x - size / 2, y - size / 2, size, size);
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y + size / 6, size / 3, Math.PI, 0);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x, y + size / 6);
      ctx.lineTo(x + size / 4, y - size / 4);
      ctx.stroke();
      ctx.restore();
      return;
    }

    // Control dots for controlled gates
    if ('control' in gate && 'target' in gate) {
      const isControl = this.getGateQubits(gate)[0] === Math.floor((y - options.padding.top) / options.cellHeight);
      if (gate.type === 'cnot' || gate.type === 'cx') {
        if (isControl) {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(x, y, 6, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
          return;
        } else {
          // Target X symbol
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, size / 2.5, 0, Math.PI * 2);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(x, y - size / 2.5);
          ctx.lineTo(x, y + size / 2.5);
          ctx.stroke();
          ctx.restore();
          return;
        }
      }
    }

    // Regular gate box
    if (style === 'circle') {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, size / 2, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.fillStyle = color;
      ctx.fillRect(x - size / 2, y - size / 2, size, size);

      if (style === 'ibm') {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 2;
        ctx.strokeRect(x - size / 2, y - size / 2, size, size);
      }
    }

    ctx.restore();
  }

  private getGateLabel(gate: Gate): string {
    const labels: Record<string, string> = {
      h: 'H',
      x: 'X',
      y: 'Y',
      z: 'Z',
      s: 'S',
      sdg: 'S†',
      t: 'T',
      tdg: 'T†',
      rx: 'Rx',
      ry: 'Ry',
      rz: 'Rz',
      phase: 'P',
      u3: 'U',
      cnot: '',
      cx: '',
      cz: 'Z',
      cy: 'Y',
      swap: '×',
      cphase: 'P',
      crx: 'Rx',
      cry: 'Ry',
      crz: 'Rz',
      toffoli: '',
      ccx: '',
      fredkin: '×',
      cswap: '×',
      qft: 'QFT',
      iqft: 'QFT†',
    };
    return labels[gate.type] || gate.type.toUpperCase();
  }

  private drawLabels(): void {
    if (!this.circuit) return;

    const { ctx, options } = this;
    const { padding, cellHeight } = options;

    ctx.save();
    ctx.font = '12px monospace';
    ctx.fillStyle = options.textColor as string;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (let q = 0; q < this.circuit.numQubits; q++) {
      const y = padding.top + q * cellHeight + cellHeight / 2;
      ctx.fillText(`q${q}`, padding.left - 10, y);
    }

    ctx.restore();
  }

  private drawScrollIndicators(): void {
    const { ctx, options } = this;
    const maxScroll = this.getMaxScroll();

    if (this.scrollX > 0) {
      // Left arrow
      ctx.save();
      ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
      ctx.beginPath();
      ctx.moveTo(options.padding.left + 10, options.height / 2 - 10);
      ctx.lineTo(options.padding.left + 10, options.height / 2 + 10);
      ctx.lineTo(options.padding.left, options.height / 2);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    if (this.scrollX < maxScroll) {
      // Right arrow
      ctx.save();
      ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
      ctx.beginPath();
      ctx.moveTo(options.width - options.padding.right - 10, options.height / 2 - 10);
      ctx.lineTo(options.width - options.padding.right - 10, options.height / 2 + 10);
      ctx.lineTo(options.width - options.padding.right, options.height / 2);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
  }
}
