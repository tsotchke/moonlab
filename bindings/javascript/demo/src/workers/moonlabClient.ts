export type WorkerGate = {
  type: string;
  qubit: number;
  controlQubit?: number;
  controlQubit2?: number;
  angle?: number;
};

type WorkerResult<T> = {
  id: number;
  ok: boolean;
  result?: T;
  error?: string;
};

type PendingRequest = {
  resolve: (value: any) => void;
  reject: (error: Error) => void;
};

export type MoonlabWorkerInitStatus = {
  ready: boolean;
  webgpu?: {
    available: boolean;
    nativeAccelerated: boolean;
    reason: string;
  };
};

let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, PendingRequest>();
let initPromise: Promise<void> | null = null;
let initStatus: MoonlabWorkerInitStatus | null = null;

const getWorker = (): Worker => {
  if (worker) return worker;
  if (typeof window === 'undefined') {
    throw new Error('Moonlab worker is only available in the browser');
  }
  const url = new URL('moonlab-worker.js', window.location.href);
  worker = new Worker(url, { type: 'classic' });
  worker.onmessage = (event: MessageEvent<WorkerResult<any>>) => {
    const message = event.data;
    const handler = pending.get(message.id);
    if (!handler) return;
    pending.delete(message.id);
    if (message.ok) {
      handler.resolve(message.result);
    } else {
      handler.reject(new Error(message.error || 'Worker error'));
    }
  };
  worker.onerror = (event) => {
    pending.forEach(({ reject }) => reject(new Error(event.message)));
    pending.clear();
  };
  return worker;
};

const callWorker = async <T>(
  type: string,
  payload?: any,
  transfer?: Transferable[]
): Promise<T> => {
  const instance = getWorker();
  const id = nextId++;
  const response = new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve, reject });
  });
  instance.postMessage({ id, type, payload }, transfer || []);
  return response;
};

export const ensureMoonlabWorker = async (): Promise<void> => {
  if (!initPromise) {
    initPromise = callWorker<MoonlabWorkerInitStatus>('init').then((status) => {
      initStatus = status;
      if (status.webgpu) {
        console.info(
          `[moonlab-worker] webgpu available=${status.webgpu.available} native=${status.webgpu.nativeAccelerated} reason=${status.webgpu.reason}`
        );
      }
    });
  }
  return initPromise;
};

export const getMoonlabWorkerInitStatus = (): MoonlabWorkerInitStatus | null => initStatus;

export const runCircuitInWorker = async (payload: {
  numQubits: number;
  gates: WorkerGate[];
}): Promise<{
  probabilities: Float64Array;
  warnings: string[];
  backend?: 'cpu' | 'webgpu';
  nativeAccelerated?: boolean;
}> => {
  const result = await callWorker<{
    probabilities: Float64Array;
    warnings: string[];
    backend?: 'cpu' | 'webgpu';
    nativeAccelerated?: boolean;
  }>(
    'runCircuit',
    payload
  );
  return result;
};

export const probabilitiesFromAmplitudes = async (payload: {
  numQubits: number;
  amplitudes: Float64Array;
}): Promise<Float64Array> => {
  const result = await callWorker<{ probabilities: Float64Array }>(
    'probabilitiesFromAmplitudes',
    payload,
    [payload.amplitudes.buffer]
  );
  return result.probabilities;
};

export const dmrgWeightsInWorker = async (payload: {
  numSites: number;
  g: number;
}): Promise<{
  weights: Float64Array;
  energy?: number;
  variance?: number;
  elapsedMs: number;
}> => {
  const result = await callWorker<{
    weights: Float64Array;
    energy?: number;
    variance?: number;
    elapsedMs: number;
  }>('dmrgWeights', payload);
  return result;
};
