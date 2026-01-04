/**
 * Amplitude Bar Chart Visualization (Canvas 2D)
 *
 * Renders quantum state amplitudes as a bar chart with phase-as-hue coloring.
 * Each bar represents a basis state's probability, colored by its phase.
 */

import type { Complex } from '@moonlab/quantum-core';
import type {
  BaseVisualizationOptions,
  ColorSchemeOptions,
  AmplitudeInfo,
  EventListener,
  InteractionEvent,
} from '../types';
import {
  amplitudesToInfo,
  amplitudeToColor,
  easeInOutCubic,
  lerp,
  clamp,
  rgbToHex,
  hslToRgb,
  phaseToHue,
} from '../types';

// ============================================================================
// Types
// ============================================================================

export interface AmplitudeBarsOptions
  extends BaseVisualizationOptions,
    ColorSchemeOptions {
  /**
   * Show basis state labels (default: true for ≤8 qubits)
   */
  showLabels?: boolean;

  /**
   * Show probability values on bars (default: true)
   */
  showValues?: boolean;

  /**
   * Show phase legend (default: true)
   */
  showPhaseLegend?: boolean;

  /**
   * Minimum probability to display (default: 0.001)
   */
  minProbability?: number;

  /**
   * Maximum number of bars to display (default: 64)
   */
  maxBars?: number;

  /**
   * Sort bars by probability (default: false)
   */
  sortByProbability?: boolean;

  /**
   * Bar spacing as fraction of bar width (default: 0.2)
   */
  barSpacing?: number;

  /**
   * Use monochrome bars (no phase coloring) (default: false)
   */
  monochrome?: boolean;

  /**
   * Orientation (default: 'vertical')
   */
  orientation?: 'vertical' | 'horizontal';

  /**
   * Padding around chart (default: { top: 40, right: 20, bottom: 60, left: 50 })
   */
  padding?: { top: number; right: number; bottom: number; left: number };
}

interface AnimationState {
  from: AmplitudeInfo[];
  to: AmplitudeInfo[];
  startTime: number;
  duration: number;
}

// ============================================================================
// AmplitudeBars Class
// ============================================================================

export class AmplitudeBars {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private options: Required<AmplitudeBarsOptions>;
  private amplitudes: AmplitudeInfo[] = [];
  private numQubits: number = 0;
  private animation: AnimationState | null = null;
  private animationFrame: number | null = null;
  private hoveredBar: number = -1;
  private listeners: Map<string, Set<EventListener>> = new Map();

  constructor(
    canvas: HTMLCanvasElement | string,
    options: AmplitudeBarsOptions = {}
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
      width: options.width ?? 600,
      height: options.height ?? 300,
      backgroundColor: options.backgroundColor ?? '#ffffff',
      animated: options.animated ?? true,
      animationDuration: options.animationDuration ?? 300,
      pixelRatio: options.pixelRatio ?? (typeof window !== 'undefined' ? window.devicePixelRatio : 1),
      scheme: options.scheme ?? 'default',
      primaryColor: options.primaryColor ?? '#4f46e5',
      secondaryColor: options.secondaryColor ?? '#06b6d4',
      textColor: options.textColor ?? '#1f2937',
      gridColor: options.gridColor ?? '#e5e7eb',
      showLabels: options.showLabels ?? true,
      showValues: options.showValues ?? true,
      showPhaseLegend: options.showPhaseLegend ?? true,
      minProbability: options.minProbability ?? 0.001,
      maxBars: options.maxBars ?? 64,
      sortByProbability: options.sortByProbability ?? false,
      barSpacing: options.barSpacing ?? 0.2,
      monochrome: options.monochrome ?? false,
      orientation: options.orientation ?? 'vertical',
      padding: options.padding ?? { top: 40, right: 20, bottom: 60, left: 50 },
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
   * Set state from complex amplitudes
   */
  setState(amplitudes: Complex[], numQubits: number): void {
    const newAmplitudes = amplitudesToInfo(amplitudes, numQubits);
    this.numQubits = numQubits;

    if (this.options.animated && this.amplitudes.length === newAmplitudes.length) {
      this.animation = {
        from: this.amplitudes.map((a) => ({ ...a })),
        to: newAmplitudes,
        startTime: performance.now(),
        duration: this.options.animationDuration,
      };
      this.animate();
    } else {
      this.amplitudes = newAmplitudes;
      this.animation = null;
      this.render();
    }
  }

  /**
   * Set state from probability distribution
   */
  setProbabilities(probabilities: number[] | Float64Array, numQubits: number): void {
    const amplitudes: Complex[] = Array.from(probabilities).map((p) => ({
      real: Math.sqrt(p),
      imag: 0,
    }));
    this.setState(amplitudes, numQubits);
  }

  /**
   * Update options
   */
  setOptions(options: Partial<AmplitudeBarsOptions>): void {
    Object.assign(this.options, options);
    if (options.width || options.height || options.pixelRatio) {
      this.setupCanvas();
    }
    this.render();
  }

  /**
   * Get displayed amplitudes
   */
  getAmplitudes(): AmplitudeInfo[] {
    return this.getDisplayedBars();
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
    if (this.animationFrame !== null) {
      cancelAnimationFrame(this.animationFrame);
    }
    this.canvas.removeEventListener('mousemove', this.onMouseMove);
    this.canvas.removeEventListener('click', this.onClick);
    this.canvas.removeEventListener('mouseleave', this.onMouseLeave);
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
    this.onMouseLeave = this.onMouseLeave.bind(this);

    this.canvas.addEventListener('mousemove', this.onMouseMove);
    this.canvas.addEventListener('click', this.onClick);
    this.canvas.addEventListener('mouseleave', this.onMouseLeave);
  }

  private getBarAtPosition(x: number, y: number): number {
    const bars = this.getDisplayedBars();
    if (bars.length === 0) return -1;

    const { padding, orientation, width, height } = this.options;
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    if (orientation === 'vertical') {
      if (y < padding.top || y > height - padding.bottom) return -1;
      if (x < padding.left || x > width - padding.right) return -1;

      const barWidth = chartWidth / bars.length;
      const index = Math.floor((x - padding.left) / barWidth);
      return index >= 0 && index < bars.length ? index : -1;
    } else {
      if (x < padding.left || x > width - padding.right) return -1;
      if (y < padding.top || y > height - padding.bottom) return -1;

      const barHeight = chartHeight / bars.length;
      const index = Math.floor((y - padding.top) / barHeight);
      return index >= 0 && index < bars.length ? index : -1;
    }
  }

  private onMouseMove(e: MouseEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const barIndex = this.getBarAtPosition(x, y);

    if (barIndex !== this.hoveredBar) {
      this.hoveredBar = barIndex;
      this.render();

      if (barIndex >= 0) {
        const bars = this.getDisplayedBars();
        const event: InteractionEvent = {
          type: 'hover',
          point: { x, y },
          basisState: bars[barIndex].index,
        };
        this.emit('hover', event);
      }
    }
  }

  private onClick(e: MouseEvent): void {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const barIndex = this.getBarAtPosition(x, y);

    if (barIndex >= 0) {
      const bars = this.getDisplayedBars();
      const event: InteractionEvent = {
        type: 'click',
        point: { x, y },
        basisState: bars[barIndex].index,
      };
      this.emit('click', event);
    }
  }

  private onMouseLeave(): void {
    if (this.hoveredBar !== -1) {
      this.hoveredBar = -1;
      this.render();
    }
  }

  private emit(event: string, data: unknown): void {
    this.listeners.get(event)?.forEach((cb) => cb(data));
  }

  private getDisplayedBars(): AmplitudeInfo[] {
    let bars = this.amplitudes.filter(
      (a) => a.probability >= this.options.minProbability
    );

    if (this.options.sortByProbability) {
      bars = bars.sort((a, b) => b.probability - a.probability);
    }

    if (bars.length > this.options.maxBars) {
      bars = bars.slice(0, this.options.maxBars);
    }

    return bars;
  }

  private animate(): void {
    if (!this.animation) return;

    const now = performance.now();
    const elapsed = now - this.animation.startTime;
    const t = clamp(elapsed / this.animation.duration, 0, 1);
    const eased = easeInOutCubic(t);

    // Interpolate amplitudes
    this.amplitudes = this.animation.to.map((to, i) => {
      const from = this.animation!.from[i];
      return {
        ...to,
        probability: lerp(from.probability, to.probability, eased),
        phase: lerp(from.phase, to.phase, eased),
        amplitude: {
          real: lerp(from.amplitude.real, to.amplitude.real, eased),
          imag: lerp(from.amplitude.imag, to.amplitude.imag, eased),
        },
      };
    });

    this.render();

    if (t < 1) {
      this.animationFrame = requestAnimationFrame(() => this.animate());
    } else {
      this.animation = null;
      this.emit('animationEnd', this.amplitudes);
    }
  }

  private render(): void {
    const { ctx, options } = this;
    const { width, height, padding } = options;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Background
    if (options.backgroundColor !== 'transparent') {
      ctx.fillStyle = options.backgroundColor as string;
      ctx.fillRect(0, 0, width, height);
    }

    const bars = this.getDisplayedBars();
    if (bars.length === 0) {
      this.drawEmptyState();
      return;
    }

    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Draw grid
    this.drawGrid(chartWidth, chartHeight);

    // Draw bars
    this.drawBars(bars, chartWidth, chartHeight);

    // Draw labels
    if (options.showLabels) {
      this.drawLabels(bars, chartWidth, chartHeight);
    }

    // Draw phase legend
    if (options.showPhaseLegend && !options.monochrome) {
      this.drawPhaseLegend();
    }
  }

  private drawEmptyState(): void {
    const { ctx, options } = this;

    ctx.save();
    ctx.font = '14px sans-serif';
    ctx.fillStyle = options.textColor as string;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('No state data', options.width / 2, options.height / 2);
    ctx.restore();
  }

  private drawGrid(chartWidth: number, chartHeight: number): void {
    const { ctx, options } = this;
    const { padding } = options;

    ctx.save();
    ctx.strokeStyle = options.gridColor as string;
    ctx.lineWidth = 1;

    // Draw horizontal grid lines
    const ySteps = [0, 0.25, 0.5, 0.75, 1];
    ctx.font = '11px sans-serif';
    ctx.fillStyle = options.textColor as string;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (const step of ySteps) {
      const y = padding.top + chartHeight * (1 - step);

      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + chartWidth, y);
      ctx.stroke();

      ctx.fillText(`${(step * 100).toFixed(0)}%`, padding.left - 8, y);
    }

    // Draw axes
    ctx.strokeStyle = options.textColor as string;
    ctx.lineWidth = 1.5;

    // Y axis
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + chartHeight);
    ctx.stroke();

    // X axis
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top + chartHeight);
    ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
    ctx.stroke();

    // Y axis label
    ctx.save();
    ctx.translate(15, padding.top + chartHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Probability', 0, 0);
    ctx.restore();

    ctx.restore();
  }

  private drawBars(
    bars: AmplitudeInfo[],
    chartWidth: number,
    chartHeight: number
  ): void {
    const { ctx, options } = this;
    const { padding, barSpacing, monochrome } = options;

    const totalBarWidth = chartWidth / bars.length;
    const barWidth = totalBarWidth * (1 - barSpacing);
    const gap = totalBarWidth * barSpacing;

    const maxProb = Math.max(...bars.map((b) => b.probability), 1);

    for (let i = 0; i < bars.length; i++) {
      const bar = bars[i];
      const x = padding.left + i * totalBarWidth + gap / 2;
      const barHeight = (bar.probability / maxProb) * chartHeight;
      const y = padding.top + chartHeight - barHeight;

      // Get bar color
      let fillColor: string;
      if (monochrome) {
        fillColor = options.primaryColor as string;
      } else {
        const rgba = amplitudeToColor(bar.amplitude, maxProb);
        fillColor = `rgba(${rgba[0]}, ${rgba[1]}, ${rgba[2]}, ${rgba[3]})`;
      }

      // Highlight hovered bar
      ctx.save();
      if (i === this.hoveredBar) {
        ctx.shadowColor = 'rgba(0, 0, 0, 0.3)';
        ctx.shadowBlur = 10;
        ctx.shadowOffsetY = 2;
      }

      // Draw bar
      ctx.fillStyle = fillColor;
      ctx.fillRect(x, y, barWidth, barHeight);

      // Draw bar border
      ctx.strokeStyle = i === this.hoveredBar ? '#000' : 'rgba(0, 0, 0, 0.2)';
      ctx.lineWidth = i === this.hoveredBar ? 2 : 1;
      ctx.strokeRect(x, y, barWidth, barHeight);

      ctx.restore();

      // Draw probability value
      if (options.showValues && bar.probability > 0.01) {
        ctx.save();
        ctx.font = '10px sans-serif';
        ctx.fillStyle = options.textColor as string;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';

        const probText = (bar.probability * 100).toFixed(1) + '%';
        ctx.fillText(probText, x + barWidth / 2, y - 4);
        ctx.restore();
      }
    }
  }

  private drawLabels(
    bars: AmplitudeInfo[],
    chartWidth: number,
    chartHeight: number
  ): void {
    const { ctx, options } = this;
    const { padding, barSpacing } = options;

    const totalBarWidth = chartWidth / bars.length;
    const barWidth = totalBarWidth * (1 - barSpacing);
    const gap = totalBarWidth * barSpacing;

    ctx.save();
    ctx.font = '10px monospace';
    ctx.fillStyle = options.textColor as string;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    for (let i = 0; i < bars.length; i++) {
      const bar = bars[i];
      const x = padding.left + i * totalBarWidth + gap / 2 + barWidth / 2;
      const y = padding.top + chartHeight + 8;

      // Rotate labels if there are many bars
      if (bars.length > 16) {
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(-Math.PI / 4);
        ctx.textAlign = 'right';
        ctx.fillText(`|${bar.bitString}⟩`, 0, 0);
        ctx.restore();
      } else {
        ctx.fillText(`|${bar.bitString}⟩`, x, y);
      }
    }

    ctx.restore();
  }

  private drawPhaseLegend(): void {
    const { ctx, options } = this;
    const legendWidth = 150;
    const legendHeight = 12;
    const legendX = options.width - options.padding.right - legendWidth;
    const legendY = 12;

    ctx.save();

    // Draw gradient
    const gradient = ctx.createLinearGradient(legendX, 0, legendX + legendWidth, 0);
    for (let i = 0; i <= 10; i++) {
      const hue = (i / 10) * 360;
      const rgb = hslToRgb([hue, 80, 50]);
      gradient.addColorStop(i / 10, rgbToHex(rgb));
    }

    ctx.fillStyle = gradient;
    ctx.fillRect(legendX, legendY, legendWidth, legendHeight);

    ctx.strokeStyle = options.gridColor as string;
    ctx.strokeRect(legendX, legendY, legendWidth, legendHeight);

    // Draw labels
    ctx.font = '10px sans-serif';
    ctx.fillStyle = options.textColor as string;
    ctx.textBaseline = 'top';

    ctx.textAlign = 'left';
    ctx.fillText('-π', legendX, legendY + legendHeight + 4);

    ctx.textAlign = 'center';
    ctx.fillText('0', legendX + legendWidth / 2, legendY + legendHeight + 4);

    ctx.textAlign = 'right';
    ctx.fillText('π', legendX + legendWidth, legendY + legendHeight + 4);

    ctx.textAlign = 'center';
    ctx.fillText('Phase', legendX + legendWidth / 2, legendY - 12);

    ctx.restore();
  }
}
