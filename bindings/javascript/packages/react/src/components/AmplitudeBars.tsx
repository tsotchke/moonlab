/**
 * AmplitudeBars React Component
 *
 * Displays quantum state amplitudes as a colorful bar chart.
 */

import React, { useRef, useEffect } from 'react';
import {
  AmplitudeBars as AmplitudeBarsViz,
  type AmplitudeBarsOptions,
} from '@moonlab/quantum-viz';
import type { Complex } from '@moonlab/quantum-core';

export interface AmplitudeBarsProps extends Omit<AmplitudeBarsOptions, 'width' | 'height'> {
  /**
   * Width in pixels (default: 600)
   */
  width?: number;

  /**
   * Height in pixels (default: 300)
   */
  height?: number;

  /**
   * Complex amplitudes of the quantum state
   */
  amplitudes?: Complex[];

  /**
   * Probabilities (alternative to amplitudes)
   */
  probabilities?: number[] | Float64Array;

  /**
   * Number of qubits (required if using probabilities)
   */
  numQubits?: number;

  /**
   * Additional CSS class name
   */
  className?: string;

  /**
   * Inline styles
   */
  style?: React.CSSProperties;

  /**
   * Callback when a bar is clicked
   */
  onBarClick?: (basisState: number, bitString: string) => void;

  /**
   * Callback when a bar is hovered
   */
  onBarHover?: (basisState: number | null, bitString: string | null) => void;
}

/**
 * AmplitudeBars React Component
 *
 * @example
 * ```tsx
 * import { AmplitudeBars } from '@moonlab/quantum-react';
 * import { useQuantumState } from '@moonlab/quantum-react';
 *
 * function App() {
 *   const { amplitudes, numQubits } = useQuantumState({ numQubits: 3 });
 *
 *   return (
 *     <AmplitudeBars
 *       amplitudes={amplitudes}
 *       numQubits={numQubits}
 *       width={600}
 *       height={300}
 *       showPhaseLegend
 *     />
 *   );
 * }
 * ```
 */
export function AmplitudeBars({
  width = 600,
  height = 300,
  amplitudes,
  probabilities,
  numQubits,
  className,
  style,
  onBarClick,
  onBarHover,
  ...options
}: AmplitudeBarsProps): React.ReactElement {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const vizRef = useRef<AmplitudeBarsViz | null>(null);

  // Calculate numQubits from data if not provided
  const effectiveNumQubits = numQubits ?? (
    amplitudes ? Math.log2(amplitudes.length) :
    probabilities ? Math.log2(probabilities.length) :
    0
  );

  // Initialize visualization
  useEffect(() => {
    if (!canvasRef.current) return;

    vizRef.current = new AmplitudeBarsViz(canvasRef.current, {
      width,
      height,
      ...options,
    });

    // Setup click listener
    if (onBarClick) {
      vizRef.current.on('click', (event: { basisState?: number }) => {
        if (event.basisState !== undefined) {
          const bitString = event.basisState.toString(2).padStart(effectiveNumQubits, '0');
          onBarClick(event.basisState, bitString);
        }
      });
    }

    // Setup hover listener
    if (onBarHover) {
      vizRef.current.on('hover', (event: { basisState?: number }) => {
        if (event.basisState !== undefined) {
          const bitString = event.basisState.toString(2).padStart(effectiveNumQubits, '0');
          onBarHover(event.basisState, bitString);
        } else {
          onBarHover(null, null);
        }
      });
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
    if (amplitudes && vizRef.current && effectiveNumQubits > 0) {
      vizRef.current.setState(amplitudes, effectiveNumQubits);
    }
  }, [amplitudes, effectiveNumQubits]);

  // Update state from probabilities
  useEffect(() => {
    if (probabilities && !amplitudes && vizRef.current && effectiveNumQubits > 0) {
      vizRef.current.setProbabilities(probabilities, effectiveNumQubits);
    }
  }, [probabilities, amplitudes, effectiveNumQubits]);

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
