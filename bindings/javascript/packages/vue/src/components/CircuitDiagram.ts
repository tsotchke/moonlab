/**
 * CircuitDiagram Vue Component
 *
 * Displays a quantum circuit as a visual diagram.
 */

import {
  defineComponent,
  ref,
  onMounted,
  onUnmounted,
  watch,
  type PropType,
  h,
} from 'vue';
import {
  CircuitDiagram as CircuitDiagramViz,
  type CircuitDiagramOptions,
} from '@moonlab/quantum-viz';
import { Circuit } from '@moonlab/quantum-core';

export const CircuitDiagram = defineComponent({
  name: 'CircuitDiagram',

  props: {
    /**
     * Width in pixels
     */
    width: {
      type: Number,
      default: 800,
    },

    /**
     * Height in pixels
     */
    height: {
      type: Number,
      default: 400,
    },

    /**
     * The circuit to display
     */
    circuit: {
      type: Object as PropType<Circuit>,
      default: undefined,
    },

    /**
     * Circuit as JSON (alternative to circuit prop)
     */
    circuitJson: {
      type: Object as PropType<ReturnType<Circuit['toJSON']>>,
      default: undefined,
    },

    /**
     * Show qubit labels
     */
    showLabels: {
      type: Boolean,
      default: true,
    },

    /**
     * Show gate names
     */
    showGateNames: {
      type: Boolean,
      default: true,
    },

    /**
     * Gate style
     */
    gateStyle: {
      type: String as PropType<'box' | 'circle' | 'ibm'>,
      default: 'box',
    },

    /**
     * Enable horizontal scrolling
     */
    scrollable: {
      type: Boolean,
      default: true,
    },
  },

  emits: ['gateClick', 'gateHover'],

  setup(props, { emit }) {
    const canvasRef = ref<HTMLCanvasElement | null>(null);
    let viz: CircuitDiagramViz | null = null;

    onMounted(() => {
      if (!canvasRef.value) return;

      viz = new CircuitDiagramViz(canvasRef.value, {
        width: props.width,
        height: props.height,
        showLabels: props.showLabels,
        showGateNames: props.showGateNames,
        gateStyle: props.gateStyle,
        scrollable: props.scrollable,
      });

      viz.on('click', (event: { gateIndex?: number }) => {
        if (event.gateIndex !== undefined) {
          emit('gateClick', event.gateIndex);
        }
      });

      viz.on('hover', (event: { gateIndex?: number }) => {
        emit('gateHover', event.gateIndex ?? null);
      });

      // Set initial circuit
      if (props.circuit) {
        viz.setCircuit(props.circuit);
      } else if (props.circuitJson) {
        viz.setCircuit(Circuit.fromJSON(props.circuitJson));
      }
    });

    onUnmounted(() => {
      viz?.dispose();
      viz = null;
    });

    // Watch for prop changes
    watch(
      () => [props.width, props.height],
      () => {
        viz?.setOptions({ width: props.width, height: props.height });
      }
    );

    watch(
      () => props.circuit,
      (newVal) => {
        if (newVal && viz) {
          viz.setCircuit(newVal);
        }
      },
      { deep: true }
    );

    watch(
      () => props.circuitJson,
      (newVal) => {
        if (newVal && !props.circuit && viz) {
          viz.setCircuit(Circuit.fromJSON(newVal));
        }
      },
      { deep: true }
    );

    return () =>
      h('canvas', {
        ref: canvasRef,
        style: { display: 'block' },
      });
  },
});
