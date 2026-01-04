/**
 * CircuitDiagram React Component
 *
 * Displays a quantum circuit as a visual diagram.
 */

import React, { useRef, useEffect } from 'react';
import {
  CircuitDiagram as CircuitDiagramViz,
  type CircuitDiagramOptions,
} from '@moonlab/quantum-viz';
import { Circuit } from '@moonlab/quantum-core';

export interface CircuitDiagramProps extends Omit<CircuitDiagramOptions, 'width' | 'height'> {
  /**
   * Width in pixels (default: 800)
   */
  width?: number;

  /**
   * Height in pixels (default: 400)
   */
  height?: number;

  /**
   * The circuit to display
   */
  circuit?: Circuit;

  /**
   * Circuit as JSON (alternative to circuit prop)
   */
  circuitJson?: ReturnType<Circuit['toJSON']>;

  /**
   * Additional CSS class name
   */
  className?: string;

  /**
   * Inline styles
   */
  style?: React.CSSProperties;

  /**
   * Callback when a gate is clicked
   */
  onGateClick?: (gateIndex: number) => void;

  /**
   * Callback when a gate is hovered
   */
  onGateHover?: (gateIndex: number | null) => void;
}

/**
 * CircuitDiagram React Component
 *
 * @example
 * ```tsx
 * import { CircuitDiagram } from '@moonlab/quantum-react';
 * import { useCircuit } from '@moonlab/quantum-react';
 *
 * function App() {
 *   const { circuit, addGate } = useCircuit({ numQubits: 3 });
 *
 *   return (
 *     <div>
 *       <CircuitDiagram circuit={circuit} width={600} height={200} />
 *       <button onClick={() => addGate('h', 0)}>Add H(0)</button>
 *     </div>
 *   );
 * }
 * ```
 */
export function CircuitDiagram({
  width = 800,
  height = 400,
  circuit,
  circuitJson,
  className,
  style,
  onGateClick,
  onGateHover,
  ...options
}: CircuitDiagramProps): React.ReactElement {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const vizRef = useRef<CircuitDiagramViz | null>(null);

  // Initialize visualization
  useEffect(() => {
    if (!canvasRef.current) return;

    vizRef.current = new CircuitDiagramViz(canvasRef.current, {
      width,
      height,
      ...options,
    });

    // Setup click listener
    if (onGateClick) {
      vizRef.current.on('click', (event: { gateIndex?: number }) => {
        if (event.gateIndex !== undefined) {
          onGateClick(event.gateIndex);
        }
      });
    }

    // Setup hover listener
    if (onGateHover) {
      vizRef.current.on('hover', (event: { gateIndex?: number }) => {
        onGateHover(event.gateIndex ?? null);
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

  // Update circuit when it changes
  useEffect(() => {
    if (circuit && vizRef.current) {
      vizRef.current.setCircuit(circuit);
    }
  }, [circuit]);

  // Update from JSON when it changes
  useEffect(() => {
    if (circuitJson && !circuit && vizRef.current) {
      vizRef.current.setCircuit(Circuit.fromJSON(circuitJson));
    }
  }, [circuitJson, circuit]);

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
