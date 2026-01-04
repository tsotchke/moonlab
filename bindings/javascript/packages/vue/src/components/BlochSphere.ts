/**
 * BlochSphere Vue Component
 *
 * Displays a single qubit state on an interactive Bloch sphere.
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
  BlochSphere as BlochSphereViz,
  type BlochSphereOptions,
} from '@moonlab/quantum-viz';
import type { Complex } from '@moonlab/quantum-core';

export interface BlochSphereEmits {
  (e: 'stateChange', state: { theta: number; phi: number; x: number; y: number; z: number }): void;
}

export const BlochSphere = defineComponent({
  name: 'BlochSphere',

  props: {
    /**
     * Width in pixels
     */
    width: {
      type: Number,
      default: 400,
    },

    /**
     * Height in pixels
     */
    height: {
      type: Number,
      default: 400,
    },

    /**
     * Amplitudes [alpha, beta] for state |psi⟩ = alpha|0⟩ + beta|1⟩
     */
    amplitudes: {
      type: Array as PropType<[Complex, Complex]>,
      default: undefined,
    },

    /**
     * Bloch angles (theta, phi) as alternative to amplitudes
     */
    angles: {
      type: Object as PropType<{ theta: number; phi: number }>,
      default: undefined,
    },

    /**
     * Cartesian coordinates (x, y, z) as alternative to amplitudes
     */
    cartesian: {
      type: Object as PropType<{ x: number; y: number; z: number }>,
      default: undefined,
    },

    /**
     * Show axis labels
     */
    showLabels: {
      type: Boolean,
      default: true,
    },

    /**
     * Show grid lines
     */
    showGrid: {
      type: Boolean,
      default: true,
    },

    /**
     * Enable mouse drag rotation
     */
    draggable: {
      type: Boolean,
      default: true,
    },

    /**
     * Show state vector
     */
    showVector: {
      type: Boolean,
      default: true,
    },

    /**
     * Enable animations
     */
    animated: {
      type: Boolean,
      default: true,
    },

    /**
     * Vector color
     */
    vectorColor: {
      type: String,
      default: '#ef4444',
    },
  },

  emits: ['stateChange'],

  setup(props, { emit }) {
    const canvasRef = ref<HTMLCanvasElement | null>(null);
    let viz: BlochSphereViz | null = null;

    onMounted(() => {
      if (!canvasRef.value) return;

      viz = new BlochSphereViz(canvasRef.value, {
        width: props.width,
        height: props.height,
        showLabels: props.showLabels,
        showGrid: props.showGrid,
        showVector: props.showVector,
        draggable: props.draggable,
        animated: props.animated,
        vectorColor: props.vectorColor,
      });

      viz.on('stateChange', (state) => {
        emit('stateChange', state);
      });

      // Set initial state
      if (props.amplitudes) {
        viz.setState(props.amplitudes);
      } else if (props.angles) {
        viz.setAngles(props.angles.theta, props.angles.phi);
      } else if (props.cartesian) {
        viz.setCartesian(props.cartesian.x, props.cartesian.y, props.cartesian.z);
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
          viz.setState(newVal);
        }
      },
      { deep: true }
    );

    watch(
      () => props.angles,
      (newVal) => {
        if (newVal && viz) {
          viz.setAngles(newVal.theta, newVal.phi);
        }
      },
      { deep: true }
    );

    watch(
      () => props.cartesian,
      (newVal) => {
        if (newVal && viz) {
          viz.setCartesian(newVal.x, newVal.y, newVal.z);
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
