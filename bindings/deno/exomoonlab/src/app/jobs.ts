/**
 * Off-frame-loop work.
 *
 * `frame()` is synchronous and runs ~30 times a second; a backend call must
 * never happen inside it. This is the whole discipline: the application asks
 * for work, keeps rendering, and reads whatever the job has most recently
 * produced. It never awaits.
 *
 * Deliberately not a general scheduler. One named slot per kind of work, the
 * latest result wins, and a superseded result is dropped rather than queued --
 * if the user has already changed the circuit, last frame's amplitudes are of
 * no interest.
 */

export type JobStatus = "idle" | "running" | "done" | "failed";

export interface JobSnapshot<T> {
  readonly status: JobStatus;
  /** Result of the most recent successful run, kept while a new one runs. */
  readonly value?: T;
  readonly error?: string;
  /** Milliseconds the last completed run took. */
  readonly durationMs?: number;
  /** Monotonically increasing; lets a view tell "new result" from "same". */
  readonly generation: number;
}

const IDLE: JobSnapshot<never> = { status: "idle", generation: 0 };

/** A single named slot of asynchronous work. */
export class Job<T> {
  #snapshot: JobSnapshot<T> = IDLE as JobSnapshot<T>;
  #generation = 0;
  /** Guards against a slow earlier run overwriting a fast later one. */
  #active = 0;

  get snapshot(): JobSnapshot<T> {
    return this.#snapshot;
  }

  /** True while a run is in flight. */
  get running(): boolean {
    return this.#snapshot.status === "running";
  }

  /**
   * Starts `work`. Returns immediately. If a run is already in flight it is
   * abandoned -- its result is discarded when it lands, because something
   * newer has been asked for.
   */
  start(work: () => Promise<T>, now: () => number = () => performance.now()): void {
    const ticket = ++this.#active;
    const startedAt = now();
    this.#snapshot = { ...this.#snapshot, status: "running" };

    work().then(
      (value) => {
        if (ticket !== this.#active) return;
        this.#snapshot = {
          status: "done",
          value,
          durationMs: now() - startedAt,
          generation: ++this.#generation,
        };
      },
      (cause: unknown) => {
        if (ticket !== this.#active) return;
        this.#snapshot = {
          status: "failed",
          value: this.#snapshot.value,
          error: cause instanceof Error ? cause.message : String(cause),
          durationMs: now() - startedAt,
          generation: ++this.#generation,
        };
      },
    );
  }
}
