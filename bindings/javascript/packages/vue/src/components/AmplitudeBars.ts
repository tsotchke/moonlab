/**
 * AmplitudeBars Vue Component
 *
 * Displays quantum state amplitudes as a colorful bar chart.
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
  AmplitudeBars as AmplitudeBarsViz,
  type AmplitudeBarsOptions,
} from '@moonlab/quantum-viz';
import type { Complex } from '@moonlab/quantum-core';

export const AmplitudeBars = defineComponent({
  name: 'AmplitudeBars',

  props: {
    /**
     * Width in pixels
     */
    width: {
      type: Number,
      default: 600,
    },

    /**
     * Height in pixels
     */
    height: {
      type: Number,
      default: 300,
    },

    /**
     * Complex amplitudes of the quantum state
     */
    amplitudes: {
      type: Array as PropType<Complex[]>,
      default: undefined,
    },

    /**
     * Probabilities (alternative to amplitudes)
     */
    probabilities: {
      type: Array as PropType<number[]>,
      default: undefined,
    },

    /**
     * Number of qubits
     */
    numQubits: {
      type: Number,
      default: undefined,
    },

    /**
     * Show basis state labels
     */
    showLabels: {
      type: Boolean,
      default: true,
    },

    /**
     * Show probability values on bars
     */
    showValues: {
      type: Boolean,
      default: true,
    },

    /**
     * Show phase legend
     */
    showPhaseLegend: {
      type: Boolean,
      default: true,
    },

    /**
     * Use monochrome bars (no phase coloring)
     */
    monochrome: {
      type: Boolean,
      default: false,
    },

    /**
     * Sort bars by probability
     */
    sortByProbability: {
      type: Boolean,
      default: false,
    },

    /**
     * Enable animations
     */
    animated: {
      type: Boolean,
      default: true,
    },
  },

  emits: ['barClick', 'barHover'],

  setup(props, { emit }) {
    const canvasRef = ref<HTMLCanvasElement | null>(null);
    let viz: AmplitudeBarsViz | null = null;

    // Calculate effective numQubits
    const getNumQubits = () => {
      if (props.numQubits) return props.numQubits;
      if (props.amplitudes) return Math.log2(props.amplitudes.length);
      if (props.probabilities) return Math.log2(props.probabilities.length);
      return 0;
    };

    onMounted(() => {
      if (!canvasRef.value) return;

      viz = new AmplitudeBarsViz(canvasRef.value, {
        width: props.width,
        height: props.height,
        showLabels: props.showLabels,
        showValues: props.showValues,
        showPhaseLegend: props.showPhaseLegend,
        monochrome: props.monochrome,
        sortByProbability: props.sortByProbability,
        animated: props.animated,
      });

      viz.on('click', (event: { basisState?: number }) => {
        if (event.basisState !== undefined) {
          const bitString = event.basisState.toString(2).padStart(getNumQubits(), '0');
          emit('barClick', { basisState: event.basisState, bitString });
        }
      });

      viz.on('hover', (event: { basisState?: number }) => {
        if (event.basisState !== undefined) {
          const bitString = event.basisState.toString(2).padStart(getNumQubits(), '0');
          emit('barHover', { basisState: event.basisState, bitString });
        } else {
          emit('barHover', null);
        }
      });

      // Set initial state
      const nq = getNumQubits();
      if (props.amplitudes && nq > 0) {
        viz.setState(props.amplitudes, nq);
      } else if (props.probabilities && nq > 0) {
        viz.setProbabilities(props.probabilities, nq);
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
      () => props.amplitudes,
      (newVal) => {
        if (newVal && viz) {
          viz.setState(newVal, getNumQubits());
        }
      },
      { deep: true }
    );

    watch(
      () => props.probabilities,
      (newVal) => {
        if (newVal && !props.amplitudes && viz) {
          viz.setProbabilities(newVal, getNumQubits());
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
