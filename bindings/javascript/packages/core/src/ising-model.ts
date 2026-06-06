import { getModule } from './wasm-loader';
import type { MoonlabModule } from './memory';

export interface IsingModelOptions {
  numQubits: number;
}

/**
 * Thin JavaScript wrapper around the WASM Ising evaluator.
 */
export class IsingModel {
  private module: MoonlabModule;
  private modelPtr: number;
  private _numQubits: number;
  private _disposed = false;

  private constructor(module: MoonlabModule, modelPtr: number, numQubits: number) {
    this.module = module;
    this.modelPtr = modelPtr;
    this._numQubits = numQubits;
  }

  static async create(options: IsingModelOptions): Promise<IsingModel> {
    validateNumQubits(options.numQubits);

    const module = await getModule();
    const modelPtr = module._ising_model_create(options.numQubits);
    if (modelPtr === 0) {
      throw new Error(`Failed to create Ising model for ${options.numQubits} qubits`);
    }

    return new IsingModel(module, modelPtr, options.numQubits);
  }

  get numQubits(): number {
    return this._numQubits;
  }

  get isDisposed(): boolean {
    return this._disposed;
  }

  setCoupling(qubit1: number, qubit2: number, value: number): this {
    this.checkDisposed();
    this.checkQubit(qubit1);
    this.checkQubit(qubit2);
    if (qubit1 === qubit2) {
      throw new Error('Coupling qubits must be different');
    }
    assertFiniteNumber(value, 'coupling value');

    const result = this.module._ising_model_set_coupling(
      this.modelPtr,
      qubit1,
      qubit2,
      value
    );
    if (result !== 0) {
      throw new Error(`Failed to set Ising coupling (${qubit1}, ${qubit2})`);
    }
    return this;
  }

  setField(qubit: number, value: number): this {
    this.checkDisposed();
    this.checkQubit(qubit);
    assertFiniteNumber(value, 'field value');

    const result = this.module._ising_model_set_field(this.modelPtr, qubit, value);
    if (result !== 0) {
      throw new Error(`Failed to set Ising field for qubit ${qubit}`);
    }
    return this;
  }

  evaluate(bitstring: number): number {
    this.checkDisposed();
    validateBitstring(bitstring, this._numQubits);
    return this.module._ising_model_evaluate(this.modelPtr, BigInt(bitstring));
  }

  dispose(): void {
    if (!this._disposed) {
      this.module._ising_model_free(this.modelPtr);
      this._disposed = true;
    }
  }

  private checkDisposed(): void {
    if (this._disposed) {
      throw new Error('IsingModel has been disposed');
    }
  }

  private checkQubit(qubit: number): void {
    if (!Number.isInteger(qubit) || qubit < 0 || qubit >= this._numQubits) {
      throw new Error(`Qubit index ${qubit} out of range [0, ${this._numQubits - 1}]`);
    }
  }
}

function validateNumQubits(numQubits: number): void {
  if (!Number.isInteger(numQubits) || numQubits < 1 || numQubits > 30) {
    throw new Error('numQubits must be an integer between 1 and 30');
  }
}

function validateBitstring(bitstring: number, numQubits: number): void {
  const maxBitstring = 2 ** numQubits;
  if (
    !Number.isInteger(bitstring) ||
    bitstring < 0 ||
    bitstring >= maxBitstring ||
    !Number.isSafeInteger(bitstring)
  ) {
    throw new Error(`bitstring must be an integer between 0 and ${maxBitstring - 1}`);
  }
}

function assertFiniteNumber(value: number, label: string): void {
  if (!Number.isFinite(value)) {
    throw new Error(`${label} must be finite`);
  }
}
