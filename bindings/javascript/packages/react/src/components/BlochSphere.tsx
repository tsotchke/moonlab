/**
 * BlochSphere React Component
 *
 * Displays a single qubit state on an interactive Bloch sphere.
 */

import React, { useRef, useEffect, useCallback } from 'react';
import {
  BlochSphere as BlochSphereViz,
  type BlochSphereOptions,
} from '@moonlab/quantum-viz';
import type { Complex } from '@moonlab/quantum-core';

export interface BlochSphereProps extends Omit<BlochSphereOptions, 'width' | 'height'> {
  /**
   * Width in pixels (default: 400)
   */
  width?: number;

  /**
   * Height in pixels (default: 400)
   */
  height?: number;

  /**
   * Amplitudes [alpha, beta] for state |psi⟩ = alpha|0⟩ + beta|1⟩
   */
  amplitudes?: [Complex, Complex];

  /**
   * Bloch angles (theta, phi) as alternative to amplitudes
   */
  angles?: { theta: number; phi: number };

  /**
   * Cartesian coordinates (x, y, z) as alternative to amplitudes
   */
  cartesian?: { x: number; y: number; z: number };

  /**
   * Additional CSS class name
   */
  className?: string;

  /**
   * Inline styles
   */
  style?: React.CSSProperties;

  /**
   * Callback when state changes (from user interaction)
   */
  onStateChange?: (state: { theta: number; phi: number; x: number; y: number; z: number }) => void;
}

/**
 * BlochSphere React Component
 *
 * @example
 * ```tsx
 * import { BlochSphere } from '@moonlab/quantum-react';
 *
 * function App() {
 *   const [amps, setAmps] = useState<[Complex, Complex]>([
 *     { real: 1, imag: 0 },
 *     { real: 0, imag: 0 }
 *   ]);
 *
 *   return (
 *     <BlochSphere
 *       amplitudes={amps}
 *       width={300}
 *       height={300}
 *       showGrid
 *       draggable
 *     />
 *   );
 * }
 * ```
 */
export function BlochSphere({
  width = 400,
  height = 400,
  amplitudes,
  angles,
  cartesian,
  className,
  style,
  onStateChange,
  ...options
}: BlochSphereProps): React.ReactElement {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const vizRef = useRef<BlochSphereViz | null>(null);

  // Initialize visualization
  useEffect(() => {
    if (!canvasRef.current) return;

    vizRef.current = new BlochSphereViz(canvasRef.current, {
      width,
      height,
      ...options,
    });

    // Setup state change listener
    if (onStateChange) {
      vizRef.current.on('stateChange', onStateChange);
    }

    return () => {
      vizRef.current?.dispose();
      vizRef.current = null;
    };
  }, []);

  // Update options when they change
  useEffect(() => {
    vizRef.current?.setOptions({ width, height, ...options });
  }, [width, height, options]);

  // Update state from amplitudes
  useEffect(() => {
    if (amplitudes && vizRef.current) {
      vizRef.current.setState(amplitudes);
    }
  }, [amplitudes]);

  // Update state from angles
  useEffect(() => {
    if (angles && vizRef.current) {
      vizRef.current.setAngles(angles.theta, angles.phi);
    }
  }, [angles]);

  // Update state from cartesian
  useEffect(() => {
    if (cartesian && vizRef.current) {
      vizRef.current.setCartesian(cartesian.x, cartesian.y, cartesian.z);
    }
  }, [cartesian]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{
        display: 'block',
        ...style,
      }}
    />
  );
}
