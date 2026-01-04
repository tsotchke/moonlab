/**
 * Bloch Sphere 3D Visualization (WebGL)
 *
 * Renders a stunning 3D Bloch sphere with smooth rotation,
 * real-time lighting, and interactive controls.
 */

import type { Complex } from '@moonlab/quantum-core';
import type {
  BaseVisualizationOptions,
  ColorSchemeOptions,
  BlochState,
  EventListener,
} from '../types';
import { amplitudesToBlochState, clamp, lerp, easeInOutCubic } from '../types';

// ============================================================================
// Types
// ============================================================================

export interface BlochSphere3DOptions
  extends BaseVisualizationOptions,
    ColorSchemeOptions {
  /** Show axis labels (default: true) */
  showLabels?: boolean;
  /** Show latitude/longitude grid (default: true) */
  showGrid?: boolean;
  /** Show the state vector arrow (default: true) */
  showVector?: boolean;
  /** Enable auto-rotation (default: false) */
  autoRotate?: boolean;
  /** Auto-rotation speed (default: 0.5) */
  autoRotateSpeed?: number;
  /** Enable mouse drag rotation (default: true) */
  draggable?: boolean;
  /** Sphere opacity (default: 0.4) */
  sphereOpacity?: number;
  /** Vector color (default: #ef4444) */
  vectorColor?: string;
  /** Ambient light intensity (default: 0.3) */
  ambientLight?: number;
  /** Directional light intensity (default: 0.7) */
  directionalLight?: number;
}

interface AnimationState {
  from: BlochState;
  to: BlochState;
  startTime: number;
  duration: number;
}

// ============================================================================
// Shader Sources
// ============================================================================

const SPHERE_VERTEX_SHADER = `
  attribute vec3 aPosition;
  attribute vec3 aNormal;

  uniform mat4 uModelViewMatrix;
  uniform mat4 uProjectionMatrix;
  uniform mat3 uNormalMatrix;

  varying vec3 vNormal;
  varying vec3 vPosition;

  void main() {
    vNormal = normalize(uNormalMatrix * aNormal);
    vPosition = (uModelViewMatrix * vec4(aPosition, 1.0)).xyz;
    gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(aPosition, 1.0);
  }
`;

const SPHERE_FRAGMENT_SHADER = `
  precision mediump float;

  uniform vec3 uAmbientColor;
  uniform vec3 uDiffuseColor;
  uniform vec3 uLightDirection;
  uniform float uAmbientIntensity;
  uniform float uDiffuseIntensity;
  uniform float uOpacity;

  varying vec3 vNormal;
  varying vec3 vPosition;

  void main() {
    vec3 normal = normalize(vNormal);
    vec3 lightDir = normalize(uLightDirection);

    // Ambient
    vec3 ambient = uAmbientColor * uAmbientIntensity;

    // Diffuse (Lambertian)
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = uDiffuseColor * diff * uDiffuseIntensity;

    // Fresnel rim effect
    vec3 viewDir = normalize(-vPosition);
    float rim = 1.0 - max(dot(viewDir, normal), 0.0);
    rim = pow(rim, 3.0) * 0.3;

    vec3 finalColor = ambient + diffuse + vec3(rim);
    gl_FragColor = vec4(finalColor, uOpacity);
  }
`;

const LINE_VERTEX_SHADER = `
  attribute vec3 aPosition;
  uniform mat4 uModelViewMatrix;
  uniform mat4 uProjectionMatrix;

  void main() {
    gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(aPosition, 1.0);
  }
`;

const LINE_FRAGMENT_SHADER = `
  precision mediump float;
  uniform vec4 uColor;

  void main() {
    gl_FragColor = uColor;
  }
`;

// ============================================================================
// BlochSphere3D Class
// ============================================================================

export class BlochSphere3D {
  private canvas: HTMLCanvasElement;
  private gl: WebGLRenderingContext;
  private options: Required<BlochSphere3DOptions>;
  private state: BlochState = { theta: 0, phi: 0, x: 0, y: 0, z: 1 };
  private animation: AnimationState | null = null;
  private animationFrame: number | null = null;
  private rotationX = 0.3;
  private rotationY = -0.5;
  private isDragging = false;
  private lastMouse = { x: 0, y: 0 };
  private listeners: Map<string, Set<EventListener>> = new Map();

  // WebGL resources
  private sphereProgram: WebGLProgram | null = null;
  private lineProgram: WebGLProgram | null = null;
  private sphereBuffers: {
    position: WebGLBuffer;
    normal: WebGLBuffer;
    index: WebGLBuffer;
    indexCount: number;
  } | null = null;

  constructor(
    canvas: HTMLCanvasElement | string,
    options: BlochSphere3DOptions = {}
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

    // Get WebGL context
    const gl = this.canvas.getContext('webgl', {
      antialias: true,
      alpha: true,
      premultipliedAlpha: false,
    });
    if (!gl) {
      throw new Error('WebGL not supported');
    }
    this.gl = gl;

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
      autoRotate: options.autoRotate ?? false,
      autoRotateSpeed: options.autoRotateSpeed ?? 0.5,
      draggable: options.draggable ?? true,
      sphereOpacity: options.sphereOpacity ?? 0.4,
      vectorColor: options.vectorColor ?? '#ef4444',
      ambientLight: options.ambientLight ?? 0.3,
      directionalLight: options.directionalLight ?? 0.7,
    };

    // Initialize
    this.setupCanvas();
    this.initShaders();
    this.initGeometry();
    this.setupInteraction();
    this.render();

    if (this.options.autoRotate) {
      this.startAutoRotate();
    }
  }

  // =========================================================================
  // Public Methods
  // =========================================================================

  setState(amplitudes: [Complex, Complex]): void {
    const newState = amplitudesToBlochState(amplitudes);
    this.transitionTo(newState);
  }

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

  reset(): void {
    this.transitionTo({ theta: 0, phi: 0, x: 0, y: 0, z: 1 });
  }

  setRotation(rotationX: number, rotationY: number): void {
    this.rotationX = rotationX;
    this.rotationY = rotationY;
    this.render();
  }

  getState(): BlochState {
    return { ...this.state };
  }

  on(event: string, callback: EventListener): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }

  off(event: string, callback: EventListener): void {
    this.listeners.get(event)?.delete(callback);
  }

  dispose(): void {
    if (this.animationFrame !== null) {
      cancelAnimationFrame(this.animationFrame);
    }
    this.canvas.removeEventListener('mousedown', this.onMouseDown);
    this.canvas.removeEventListener('touchstart', this.onTouchStart);
    this.listeners.clear();

    // Clean up WebGL resources
    const { gl } = this;
    if (this.sphereBuffers) {
      gl.deleteBuffer(this.sphereBuffers.position);
      gl.deleteBuffer(this.sphereBuffers.normal);
      gl.deleteBuffer(this.sphereBuffers.index);
    }
    if (this.sphereProgram) gl.deleteProgram(this.sphereProgram);
    if (this.lineProgram) gl.deleteProgram(this.lineProgram);
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
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
  }

  private initShaders(): void {
    const { gl } = this;

    // Sphere shader
    this.sphereProgram = this.createProgram(SPHERE_VERTEX_SHADER, SPHERE_FRAGMENT_SHADER);

    // Line shader
    this.lineProgram = this.createProgram(LINE_VERTEX_SHADER, LINE_FRAGMENT_SHADER);
  }

  private createProgram(vsSource: string, fsSource: string): WebGLProgram {
    const { gl } = this;

    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vsSource);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      throw new Error(`Vertex shader error: ${gl.getShaderInfoLog(vs)}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, fsSource);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      throw new Error(`Fragment shader error: ${gl.getShaderInfoLog(fs)}`);
    }

    const program = gl.createProgram()!;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`Program link error: ${gl.getProgramInfoLog(program)}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);

    return program;
  }

  private initGeometry(): void {
    const { gl } = this;

    // Generate sphere geometry
    const latBands = 32;
    const longBands = 32;
    const radius = 1;

    const positions: number[] = [];
    const normals: number[] = [];
    const indices: number[] = [];

    for (let lat = 0; lat <= latBands; lat++) {
      const theta = (lat * Math.PI) / latBands;
      const sinTheta = Math.sin(theta);
      const cosTheta = Math.cos(theta);

      for (let lon = 0; lon <= longBands; lon++) {
        const phi = (lon * 2 * Math.PI) / longBands;
        const sinPhi = Math.sin(phi);
        const cosPhi = Math.cos(phi);

        const x = cosPhi * sinTheta;
        const y = cosTheta;
        const z = sinPhi * sinTheta;

        positions.push(radius * x, radius * y, radius * z);
        normals.push(x, y, z);
      }
    }

    for (let lat = 0; lat < latBands; lat++) {
      for (let lon = 0; lon < longBands; lon++) {
        const first = lat * (longBands + 1) + lon;
        const second = first + longBands + 1;

        indices.push(first, second, first + 1);
        indices.push(second, second + 1, first + 1);
      }
    }

    // Create buffers
    const positionBuffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);

    const normalBuffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);

    const indexBuffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);

    this.sphereBuffers = {
      position: positionBuffer,
      normal: normalBuffer,
      index: indexBuffer,
      indexCount: indices.length,
    };
  }

  private setupInteraction(): void {
    if (!this.options.draggable) return;

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
      this.animateTransition();
    } else {
      this.state = newState;
      this.animation = null;
      this.render();
    }
  }

  private animateTransition(): void {
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
      this.animationFrame = requestAnimationFrame(() => this.animateTransition());
    } else {
      this.animation = null;
      this.emit('animationEnd', this.state);
    }
  }

  private startAutoRotate(): void {
    const rotate = () => {
      if (!this.isDragging) {
        this.rotationY += this.options.autoRotateSpeed * 0.01;
        this.render();
      }
      this.animationFrame = requestAnimationFrame(rotate);
    };
    rotate();
  }

  private emit(event: string, data: unknown): void {
    this.listeners.get(event)?.forEach((cb) => cb(data));
  }

  private render(): void {
    const { gl, options } = this;

    // Clear
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    // Create matrices
    const modelViewMatrix = this.createModelViewMatrix();
    const projectionMatrix = this.createProjectionMatrix();
    const normalMatrix = this.createNormalMatrix(modelViewMatrix);

    // Draw sphere
    this.drawSphere(modelViewMatrix, projectionMatrix, normalMatrix);

    // Draw grid
    if (options.showGrid) {
      this.drawGrid(modelViewMatrix, projectionMatrix);
    }

    // Draw axes
    this.drawAxes(modelViewMatrix, projectionMatrix);

    // Draw state vector
    if (options.showVector) {
      this.drawVector(modelViewMatrix, projectionMatrix);
    }

    // Draw labels (using 2D overlay)
    if (options.showLabels) {
      this.drawLabels(modelViewMatrix, projectionMatrix);
    }
  }

  private createModelViewMatrix(): Float32Array {
    const mat = new Float32Array(16);

    // Identity
    mat[0] = mat[5] = mat[10] = mat[15] = 1;

    // Translate back
    mat[14] = -3;

    // Rotate X
    const cx = Math.cos(this.rotationX);
    const sx = Math.sin(this.rotationX);
    const rotX = new Float32Array([
      1, 0, 0, 0,
      0, cx, sx, 0,
      0, -sx, cx, 0,
      0, 0, 0, 1,
    ]);

    // Rotate Y
    const cy = Math.cos(this.rotationY);
    const sy = Math.sin(this.rotationY);
    const rotY = new Float32Array([
      cy, 0, -sy, 0,
      0, 1, 0, 0,
      sy, 0, cy, 0,
      0, 0, 0, 1,
    ]);

    // Multiply: mat = translate * rotX * rotY
    const temp = this.multiplyMatrices(mat, rotX);
    return this.multiplyMatrices(temp, rotY);
  }

  private createProjectionMatrix(): Float32Array {
    const fov = Math.PI / 4;
    const aspect = this.options.width / this.options.height;
    const near = 0.1;
    const far = 100;

    const f = 1 / Math.tan(fov / 2);
    const nf = 1 / (near - far);

    return new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, (far + near) * nf, -1,
      0, 0, 2 * far * near * nf, 0,
    ]);
  }

  private createNormalMatrix(modelView: Float32Array): Float32Array {
    // Extract 3x3 from model-view and invert-transpose
    // Simplified: just use upper-left 3x3 for rotation
    return new Float32Array([
      modelView[0], modelView[1], modelView[2],
      modelView[4], modelView[5], modelView[6],
      modelView[8], modelView[9], modelView[10],
    ]);
  }

  private multiplyMatrices(a: Float32Array, b: Float32Array): Float32Array {
    const result = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        result[i * 4 + j] =
          a[i * 4] * b[j] +
          a[i * 4 + 1] * b[j + 4] +
          a[i * 4 + 2] * b[j + 8] +
          a[i * 4 + 3] * b[j + 12];
      }
    }
    return result;
  }

  private drawSphere(
    modelView: Float32Array,
    projection: Float32Array,
    normalMat: Float32Array
  ): void {
    const { gl, sphereProgram, sphereBuffers, options } = this;
    if (!sphereProgram || !sphereBuffers) return;

    gl.useProgram(sphereProgram);

    // Position attribute
    const posLoc = gl.getAttribLocation(sphereProgram, 'aPosition');
    gl.bindBuffer(gl.ARRAY_BUFFER, sphereBuffers.position);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

    // Normal attribute
    const normLoc = gl.getAttribLocation(sphereProgram, 'aNormal');
    gl.bindBuffer(gl.ARRAY_BUFFER, sphereBuffers.normal);
    gl.enableVertexAttribArray(normLoc);
    gl.vertexAttribPointer(normLoc, 3, gl.FLOAT, false, 0, 0);

    // Uniforms
    gl.uniformMatrix4fv(gl.getUniformLocation(sphereProgram, 'uModelViewMatrix'), false, modelView);
    gl.uniformMatrix4fv(gl.getUniformLocation(sphereProgram, 'uProjectionMatrix'), false, projection);
    gl.uniformMatrix3fv(gl.getUniformLocation(sphereProgram, 'uNormalMatrix'), false, normalMat);

    gl.uniform3f(gl.getUniformLocation(sphereProgram, 'uAmbientColor'), 0.6, 0.7, 0.9);
    gl.uniform3f(gl.getUniformLocation(sphereProgram, 'uDiffuseColor'), 0.7, 0.8, 1.0);
    gl.uniform3f(gl.getUniformLocation(sphereProgram, 'uLightDirection'), 0.5, 0.7, 1.0);
    gl.uniform1f(gl.getUniformLocation(sphereProgram, 'uAmbientIntensity'), options.ambientLight);
    gl.uniform1f(gl.getUniformLocation(sphereProgram, 'uDiffuseIntensity'), options.directionalLight);
    gl.uniform1f(gl.getUniformLocation(sphereProgram, 'uOpacity'), options.sphereOpacity);

    // Draw
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, sphereBuffers.index);
    gl.drawElements(gl.TRIANGLES, sphereBuffers.indexCount, gl.UNSIGNED_SHORT, 0);
  }

  private drawGrid(modelView: Float32Array, projection: Float32Array): void {
    const { gl, lineProgram } = this;
    if (!lineProgram) return;

    gl.useProgram(lineProgram);
    gl.uniformMatrix4fv(gl.getUniformLocation(lineProgram, 'uModelViewMatrix'), false, modelView);
    gl.uniformMatrix4fv(gl.getUniformLocation(lineProgram, 'uProjectionMatrix'), false, projection);
    gl.uniform4f(gl.getUniformLocation(lineProgram, 'uColor'), 0.5, 0.5, 0.6, 0.3);

    // Draw longitude lines
    for (let lon = 0; lon < 360; lon += 30) {
      const vertices: number[] = [];
      for (let lat = -90; lat <= 90; lat += 5) {
        const theta = ((90 - lat) * Math.PI) / 180;
        const phi = (lon * Math.PI) / 180;
        vertices.push(
          Math.sin(theta) * Math.cos(phi),
          Math.cos(theta),
          Math.sin(theta) * Math.sin(phi)
        );
      }
      this.drawLineStrip(vertices);
    }

    // Draw latitude lines
    for (let lat = -60; lat <= 60; lat += 30) {
      const vertices: number[] = [];
      for (let lon = 0; lon <= 360; lon += 5) {
        const theta = ((90 - lat) * Math.PI) / 180;
        const phi = (lon * Math.PI) / 180;
        vertices.push(
          Math.sin(theta) * Math.cos(phi),
          Math.cos(theta),
          Math.sin(theta) * Math.sin(phi)
        );
      }
      this.drawLineStrip(vertices);
    }
  }

  private drawAxes(modelView: Float32Array, projection: Float32Array): void {
    const { gl, lineProgram } = this;
    if (!lineProgram) return;

    gl.useProgram(lineProgram);
    gl.uniformMatrix4fv(gl.getUniformLocation(lineProgram, 'uModelViewMatrix'), false, modelView);
    gl.uniformMatrix4fv(gl.getUniformLocation(lineProgram, 'uProjectionMatrix'), false, projection);

    // X axis (red)
    gl.uniform4f(gl.getUniformLocation(lineProgram, 'uColor'), 0.94, 0.27, 0.27, 1.0);
    this.drawLineStrip([-1.2, 0, 0, 1.2, 0, 0]);

    // Y axis (green)
    gl.uniform4f(gl.getUniformLocation(lineProgram, 'uColor'), 0.13, 0.77, 0.35, 1.0);
    this.drawLineStrip([0, -1.2, 0, 0, 1.2, 0]);

    // Z axis (blue)
    gl.uniform4f(gl.getUniformLocation(lineProgram, 'uColor'), 0.23, 0.51, 0.96, 1.0);
    this.drawLineStrip([0, 0, -1.2, 0, 0, 1.2]);
  }

  private drawVector(modelView: Float32Array, projection: Float32Array): void {
    const { gl, lineProgram, state, options } = this;
    if (!lineProgram) return;

    gl.useProgram(lineProgram);
    gl.uniformMatrix4fv(gl.getUniformLocation(lineProgram, 'uModelViewMatrix'), false, modelView);
    gl.uniformMatrix4fv(gl.getUniformLocation(lineProgram, 'uProjectionMatrix'), false, projection);

    // Parse vector color
    const hex = options.vectorColor.replace('#', '');
    const r = parseInt(hex.substring(0, 2), 16) / 255;
    const g = parseInt(hex.substring(2, 4), 16) / 255;
    const b = parseInt(hex.substring(4, 6), 16) / 255;

    gl.uniform4f(gl.getUniformLocation(lineProgram, 'uColor'), r, g, b, 1.0);

    // Draw line from origin to state
    gl.lineWidth(3);
    this.drawLineStrip([0, 0, 0, state.x, state.z, state.y]); // Swap y/z for WebGL coords
  }

  private drawLineStrip(vertices: number[]): void {
    const { gl, lineProgram } = this;
    if (!lineProgram) return;

    const buffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

    const posLoc = gl.getAttribLocation(lineProgram, 'aPosition');
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.LINE_STRIP, 0, vertices.length / 3);
    gl.deleteBuffer(buffer);
  }

  private drawLabels(modelView: Float32Array, projection: Float32Array): void {
    // Labels would require a 2D overlay canvas or text rendering
    // For simplicity, we skip WebGL text rendering
    // Users can overlay an HTML layer for labels
  }
}
