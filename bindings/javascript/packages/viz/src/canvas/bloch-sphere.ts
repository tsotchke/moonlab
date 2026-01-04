/**
 * Bloch Sphere Visualization (Canvas 2D)
 *
 * Renders a single qubit state on a 2D projection of the Bloch sphere.
 * Supports animation and interactive rotation.
 */

import type { Complex } from '@moonlab/quantum-core';
import type {
  BaseVisualizationOptions,
  ColorSchemeOptions,
  BlochState,
  Point2D,
  EventListener,
} from '../types';
import {
  amplitudesToBlochState,
  easeInOutCubic,
  lerp,
  clamp,
} from '../types';

// ============================================================================
// Types
// ============================================================================

export interface BlochSphereOptions
  extends BaseVisualizationOptions,
    ColorSchemeOptions {
  /**
   * Show axis labels (X, Y, Z, |0⟩, |1⟩) (default: true)
   */
  showLabels?: boolean;

  /**
   * Show latitude/longitude grid lines (default: true)
   */
  showGrid?: boolean;

  /**
   * Show the state vector arrow (default: true)
   */
  showVector?: boolean;

  /**
   * Show projection lines to axes (default: false)
   */
  showProjections?: boolean;

  /**
   * Show theta/phi angle labels (default: false)
   */
  showAngles?: boolean;

  /**
   * Vector color (default: derived from phase)
   */
  vectorColor?: string;

  /**
   * Sphere opacity (default: 0.3)
   */
  sphereOpacity?: number;

  /**
   * Initial rotation around Y axis in radians (default: -0.3)
   */
  rotationY?: number;

  /**
   * Initial rotation around X axis in radians (default: 0.3)
   */
  rotationX?: number;

  /**
   * Enable mouse drag rotation (default: true)
   */
  draggable?: boolean;
}

interface AnimationState {
  from: BlochState;
  to: BlochState;
  startTime: number;
  duration: number;
}

// ============================================================================
// BlochSphere Class
// ============================================================================

export class BlochSphere {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private options: Required<BlochSphereOptions>;
  private state: BlochState = { theta: 0, phi: 0, x: 0, y: 0, z: 1 };
  private animation: AnimationState | null = null;
  private animationFrame: number | null = null;
  private rotationX: number;
  private rotationY: number;
  private isDragging = false;
  private lastMouse: Point2D = { x: 0, y: 0 };
  private listeners: Map<string, Set<EventListener>> = new Map();

  constructor(
    canvas: HTMLCanvasElement | string,
    options: BlochSphereOptions = {}
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
      width: options.width ?? 400,
      height: options.height ?? 400,
      backgroundColor: options.backgroundColor ?? 'transparent',
      animated: options.animated ?? true,
      animationDuration: options.animationDuration ?? 300,
      pixelRatio: options.pixelRatio ?? (typeof window !== 'undefined' ? window.devicePixelRatio : 1),
      scheme: options.scheme ?? 'default',
      primaryColor: options.primaryColor ?? '#4f46e5',
      secondaryColor: options.secondaryColor ?? '#06b6d4',
      textColor: options.textColor ?? '#1f2937',
      gridColor: options.gridColor ?? '#d1d5db',
      showLabels: options.showLabels ?? true,
      showGrid: options.showGrid ?? true,
      showVector: options.showVector ?? true,
      showProjections: options.showProjections ?? false,
      showAngles: options.showAngles ?? false,
      vectorColor: options.vectorColor ?? '#ef4444',
      sphereOpacity: options.sphereOpacity ?? 0.3,
      rotationY: options.rotationY ?? -0.3,
      rotationX: options.rotationX ?? 0.3,
      draggable: options.draggable ?? true,
    };

    this.rotationX = this.options.rotationX;
    this.rotationY = this.options.rotationY;

    // Setup canvas
    this.setupCanvas();

    // Setup interaction
    if (this.options.draggable) {
      this.setupInteraction();
    }

    // Initial render
    this.render();
  }

  // =========================================================================
  // Public Methods
  // =========================================================================

  /**
   * Set state from complex amplitudes [α, β] where |ψ⟩ = α|0⟩ + β|1⟩
   */
  setState(amplitudes: [Complex, Complex]): void {
    const newState = amplitudesToBlochState(amplitudes);
    this.transitionTo(newState);
  }

  /**
   * Set state from Bloch angles (theta, phi)
   */
  setAngles(theta: number, phi: number): void {
    const newState: BlochState = {
      theta,
      phi,
      x: Math.sin(theta) * Math.cos(phi),
      y: Math.sin(theta) * Math.sin(phi),
      z: Math.cos(theta),
    };
    this.transitionTo(newState);
  }

  /**
   * Set state from Cartesian coordinates on Bloch sphere
   */
  setCartesian(x: number, y: number, z: number): void {
    const r = Math.sqrt(x * x + y * y + z * z);
    if (r > 0) {
      x /= r;
      y /= r;
      z /= r;
    }
    const theta = Math.acos(clamp(z, -1, 1));
    const phi = Math.atan2(y, x);
    this.transitionTo({ theta, phi, x, y, z });
  }

  /**
   * Reset to |0⟩ state (north pole)
   */
  reset(): void {
    this.transitionTo({ theta: 0, phi: 0, x: 0, y: 0, z: 1 });
  }

  /**
   * Set rotation angles for 3D view
   */
  setRotation(rotationX: number, rotationY: number): void {
    this.rotationX = rotationX;
    this.rotationY = rotationY;
    this.render();
  }

  /**
   * Update options
   */
  setOptions(options: Partial<BlochSphereOptions>): void {
    Object.assign(this.options, options);
    if (options.width || options.height || options.pixelRatio) {
      this.setupCanvas();
    }
    this.render();
  }

  /**
   * Get current Bloch state
   */
  getState(): BlochState {
    return { ...this.state };
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
    this.canvas.removeEventListener('mousedown', this.onMouseDown);
    this.canvas.removeEventListener('touchstart', this.onTouchStart);
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
    this.onMouseDown = this.onMouseDown.bind(this);
    this.onMouseMove = this.onMouseMove.bind(this);
    this.onMouseUp = this.onMouseUp.bind(this);
    this.onTouchStart = this.onTouchStart.bind(this);
    this.onTouchMove = this.onTouchMove.bind(this);
    this.onTouchEnd = this.onTouchEnd.bind(this);

    this.canvas.addEventListener('mousedown', this.onMouseDown);
    this.canvas.addEventListener('touchstart', this.onTouchStart, { passive: false });
  }

  private onMouseDown(e: MouseEvent): void {
    this.isDragging = true;
    this.lastMouse = { x: e.clientX, y: e.clientY };
    window.addEventListener('mousemove', this.onMouseMove);
    window.addEventListener('mouseup', this.onMouseUp);
  }

  private onMouseMove(e: MouseEvent): void {
    if (!this.isDragging) return;

    const dx = e.clientX - this.lastMouse.x;
    const dy = e.clientY - this.lastMouse.y;

    this.rotationY += dx * 0.01;
    this.rotationX += dy * 0.01;
    this.rotationX = clamp(this.rotationX, -Math.PI / 2, Math.PI / 2);

    this.lastMouse = { x: e.clientX, y: e.clientY };
    this.render();
  }

  private onMouseUp(): void {
    this.isDragging = false;
    window.removeEventListener('mousemove', this.onMouseMove);
    window.removeEventListener('mouseup', this.onMouseUp);
  }

  private onTouchStart(e: TouchEvent): void {
    if (e.touches.length === 1) {
      e.preventDefault();
      this.isDragging = true;
      this.lastMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      window.addEventListener('touchmove', this.onTouchMove, { passive: false });
      window.addEventListener('touchend', this.onTouchEnd);
    }
  }

  private onTouchMove(e: TouchEvent): void {
    if (!this.isDragging || e.touches.length !== 1) return;
    e.preventDefault();

    const dx = e.touches[0].clientX - this.lastMouse.x;
    const dy = e.touches[0].clientY - this.lastMouse.y;

    this.rotationY += dx * 0.01;
    this.rotationX += dy * 0.01;
    this.rotationX = clamp(this.rotationX, -Math.PI / 2, Math.PI / 2);

    this.lastMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    this.render();
  }

  private onTouchEnd(): void {
    this.isDragging = false;
    window.removeEventListener('touchmove', this.onTouchMove);
    window.removeEventListener('touchend', this.onTouchEnd);
  }

  private transitionTo(newState: BlochState): void {
    if (this.options.animated && this.animation === null) {
      this.animation = {
        from: { ...this.state },
        to: newState,
        startTime: performance.now(),
        duration: this.options.animationDuration,
      };
      this.animate();
    } else {
      this.state = newState;
      this.animation = null;
      this.render();
    }
  }

  private animate(): void {
    if (!this.animation) return;

    const now = performance.now();
    const elapsed = now - this.animation.startTime;
    const t = clamp(elapsed / this.animation.duration, 0, 1);
    const eased = easeInOutCubic(t);

    this.state = {
      theta: lerp(this.animation.from.theta, this.animation.to.theta, eased),
      phi: lerp(this.animation.from.phi, this.animation.to.phi, eased),
      x: lerp(this.animation.from.x, this.animation.to.x, eased),
      y: lerp(this.animation.from.y, this.animation.to.y, eased),
      z: lerp(this.animation.from.z, this.animation.to.z, eased),
    };

    this.render();

    if (t < 1) {
      this.animationFrame = requestAnimationFrame(() => this.animate());
    } else {
      this.animation = null;
      this.emit('animationEnd', this.state);
    }
  }

  private emit(event: string, data: unknown): void {
    this.listeners.get(event)?.forEach((cb) => cb(data));
  }

  private render(): void {
    const { ctx, options } = this;
    const { width, height } = options;
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.min(width, height) * 0.35;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Background
    if (options.backgroundColor !== 'transparent') {
      ctx.fillStyle = options.backgroundColor as string;
      ctx.fillRect(0, 0, width, height);
    }

    // Draw sphere
    this.drawSphere(cx, cy, radius);

    // Draw grid
    if (options.showGrid) {
      this.drawGrid(cx, cy, radius);
    }

    // Draw axes
    this.drawAxes(cx, cy, radius);

    // Draw state vector
    if (options.showVector) {
      this.drawVector(cx, cy, radius);
    }

    // Draw labels
    if (options.showLabels) {
      this.drawLabels(cx, cy, radius);
    }

    // Draw angles
    if (options.showAngles) {
      this.drawAngles(cx, cy, radius);
    }
  }

  private project(x: number, y: number, z: number): Point2D {
    // Apply rotations
    const cosY = Math.cos(this.rotationY);
    const sinY = Math.sin(this.rotationY);
    const cosX = Math.cos(this.rotationX);
    const sinX = Math.sin(this.rotationX);

    // Rotate around Y axis
    let rx = x * cosY + z * sinY;
    let ry = y;
    let rz = -x * sinY + z * cosY;

    // Rotate around X axis
    const rry = ry * cosX - rz * sinX;
    rz = ry * sinX + rz * cosX;
    ry = rry;

    // Return 2D projection
    return { x: rx, y: -ry };
  }

  private drawSphere(cx: number, cy: number, radius: number): void {
    const { ctx, options } = this;

    // Draw filled ellipse for sphere
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(cx, cy, radius, radius, 0, 0, Math.PI * 2);

    const gradient = ctx.createRadialGradient(
      cx - radius * 0.3,
      cy - radius * 0.3,
      0,
      cx,
      cy,
      radius
    );
    gradient.addColorStop(0, `rgba(200, 200, 255, ${options.sphereOpacity})`);
    gradient.addColorStop(1, `rgba(100, 100, 200, ${options.sphereOpacity * 0.5})`);

    ctx.fillStyle = gradient;
    ctx.fill();

    ctx.strokeStyle = options.gridColor as string;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();
  }

  private drawGrid(cx: number, cy: number, radius: number): void {
    const { ctx, options } = this;

    ctx.save();
    ctx.strokeStyle = options.gridColor as string;
    ctx.lineWidth = 0.5;
    ctx.globalAlpha = 0.5;

    // Longitude lines (meridians)
    for (let lon = 0; lon < 360; lon += 30) {
      ctx.beginPath();
      for (let lat = -90; lat <= 90; lat += 5) {
        const theta = ((90 - lat) * Math.PI) / 180;
        const phi = (lon * Math.PI) / 180;

        const x = Math.sin(theta) * Math.cos(phi);
        const y = Math.sin(theta) * Math.sin(phi);
        const z = Math.cos(theta);

        const p = this.project(x, y, z);
        const px = cx + p.x * radius;
        const py = cy + p.y * radius;

        if (lat === -90) {
          ctx.moveTo(px, py);
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.stroke();
    }

    // Latitude lines (parallels)
    for (let lat = -60; lat <= 60; lat += 30) {
      ctx.beginPath();
      for (let lon = 0; lon <= 360; lon += 5) {
        const theta = ((90 - lat) * Math.PI) / 180;
        const phi = (lon * Math.PI) / 180;

        const x = Math.sin(theta) * Math.cos(phi);
        const y = Math.sin(theta) * Math.sin(phi);
        const z = Math.cos(theta);

        const p = this.project(x, y, z);
        const px = cx + p.x * radius;
        const py = cy + p.y * radius;

        if (lon === 0) {
          ctx.moveTo(px, py);
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.stroke();
    }

    ctx.restore();
  }

  private drawAxes(cx: number, cy: number, radius: number): void {
    const { ctx, options } = this;
    const axisLength = radius * 1.2;

    ctx.save();
    ctx.lineWidth = 1.5;

    // Draw each axis
    const axes = [
      { start: [-1, 0, 0], end: [1, 0, 0], color: '#ef4444' }, // X axis (red)
      { start: [0, -1, 0], end: [0, 1, 0], color: '#22c55e' }, // Y axis (green)
      { start: [0, 0, -1], end: [0, 0, 1], color: '#3b82f6' }, // Z axis (blue)
    ];

    for (const axis of axes) {
      const p1 = this.project(axis.start[0], axis.start[1], axis.start[2]);
      const p2 = this.project(axis.end[0], axis.end[1], axis.end[2]);

      ctx.beginPath();
      ctx.strokeStyle = axis.color;
      ctx.moveTo(cx + p1.x * axisLength, cy + p1.y * axisLength);
      ctx.lineTo(cx + p2.x * axisLength, cy + p2.y * axisLength);
      ctx.stroke();
    }

    ctx.restore();
  }

  private drawVector(cx: number, cy: number, radius: number): void {
    const { ctx, options, state } = this;

    const p = this.project(state.x, state.y, state.z);
    const px = cx + p.x * radius;
    const py = cy + p.y * radius;

    ctx.save();

    // Draw vector line
    ctx.beginPath();
    ctx.strokeStyle = options.vectorColor;
    ctx.lineWidth = 3;
    ctx.moveTo(cx, cy);
    ctx.lineTo(px, py);
    ctx.stroke();

    // Draw arrowhead
    const angle = Math.atan2(py - cy, px - cx);
    const arrowSize = 10;

    ctx.beginPath();
    ctx.fillStyle = options.vectorColor;
    ctx.moveTo(px, py);
    ctx.lineTo(
      px - arrowSize * Math.cos(angle - Math.PI / 6),
      py - arrowSize * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      px - arrowSize * Math.cos(angle + Math.PI / 6),
      py - arrowSize * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fill();

    // Draw state point
    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.fillStyle = options.vectorColor;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw projections if enabled
    if (options.showProjections) {
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)';
      ctx.lineWidth = 1;

      // XY plane projection
      const pxy = this.project(state.x, state.y, 0);
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(cx + pxy.x * radius, cy + pxy.y * radius);
      ctx.stroke();

      // To Z axis
      const pz = this.project(0, 0, state.z);
      ctx.beginPath();
      ctx.moveTo(cx + pxy.x * radius, cy + pxy.y * radius);
      ctx.lineTo(cx + pz.x * radius, cy + pz.y * radius);
      ctx.stroke();
    }

    ctx.restore();
  }

  private drawLabels(cx: number, cy: number, radius: number): void {
    const { ctx, options } = this;
    const labelOffset = radius * 1.3;

    ctx.save();
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = options.textColor as string;

    // Axis labels
    const labels = [
      { pos: [1, 0, 0], text: 'X', color: '#ef4444' },
      { pos: [-1, 0, 0], text: '-X', color: '#ef4444' },
      { pos: [0, 1, 0], text: 'Y', color: '#22c55e' },
      { pos: [0, -1, 0], text: '-Y', color: '#22c55e' },
      { pos: [0, 0, 1], text: '|0⟩', color: '#3b82f6' },
      { pos: [0, 0, -1], text: '|1⟩', color: '#3b82f6' },
    ];

    for (const label of labels) {
      const p = this.project(label.pos[0], label.pos[1], label.pos[2]);
      ctx.fillStyle = label.color;
      ctx.fillText(label.text, cx + p.x * labelOffset, cy + p.y * labelOffset);
    }

    ctx.restore();
  }

  private drawAngles(cx: number, cy: number, radius: number): void {
    const { ctx, options, state } = this;

    ctx.save();
    ctx.font = '12px monospace';
    ctx.fillStyle = options.textColor as string;
    ctx.textAlign = 'left';

    const thetaDeg = ((state.theta * 180) / Math.PI).toFixed(1);
    const phiDeg = ((state.phi * 180) / Math.PI).toFixed(1);

    ctx.fillText(`θ = ${thetaDeg}°`, 10, 20);
    ctx.fillText(`φ = ${phiDeg}°`, 10, 36);

    ctx.restore();
  }
}
