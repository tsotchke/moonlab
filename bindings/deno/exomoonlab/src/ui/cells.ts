/**
 * Cell primitives.
 *
 * These mirror exotui's `ShellPresentedCell` / `ShellPresentedFrame`
 * structurally rather than importing them: at 0.6.0 those types live in
 * `src/app/shell_presenter.ts`, which is not a published entrypoint, and
 * reaching into a dependency's `src/` is not something we do. TypeScript is
 * structural, so a frame built here is accepted by `runConsoleShellApp`
 * unchanged. When exotui publishes the seam, this file's types become
 * re-exports and nothing else moves.
 */

/** Red, green, blue -- 0..255. */
export type Rgb = readonly [number, number, number];

export interface Cell {
  readonly char: string;
  readonly foreground?: Rgb;
  readonly background?: Rgb;
  readonly bold?: boolean;
}

/** Row-major; every row is exactly `columns` wide. */
export type Frame = ReadonlyArray<ReadonlyArray<Cell>>;

export interface Style {
  readonly foreground?: Rgb;
  readonly background?: Rgb;
  readonly bold?: boolean;
}

/** A writable grid that composes into a {@link Frame}. */
export class Surface {
  readonly columns: number;
  readonly rows: number;
  readonly #cells: Cell[][];

  constructor(columns: number, rows: number, fill: Style = {}) {
    this.columns = Math.max(0, Math.floor(columns));
    this.rows = Math.max(0, Math.floor(rows));
    this.#cells = Array.from(
      { length: this.rows },
      () => Array.from({ length: this.columns }, () => ({ char: " ", ...fill })),
    );
  }

  set(column: number, row: number, char: string, style: Style = {}): void {
    if (row < 0 || row >= this.rows || column < 0 || column >= this.columns) return;
    this.#cells[row][column] = { char, ...style };
  }

  /** Writes text, clipped to the surface. Returns the column after the text. */
  write(column: number, row: number, text: string, style: Style = {}): number {
    let x = column;
    for (const char of text) {
      if (x >= this.columns) break;
      this.set(x, row, char, style);
      x += 1;
    }
    return x;
  }

  /** Writes text truncated to `width`, with an ellipsis when it does not fit. */
  writeFitted(column: number, row: number, text: string, width: number, style: Style = {}): void {
    const glyphs = [...text];
    if (glyphs.length <= width) {
      this.write(column, row, text, style);
      return;
    }
    if (width <= 1) {
      this.write(column, row, "…".slice(0, width), style);
      return;
    }
    this.write(column, row, `${glyphs.slice(0, width - 1).join("")}…`, style);
  }

  /** A single-line box; contents are the caller's business. */
  box(column: number, row: number, width: number, height: number, style: Style = {}): void {
    if (width < 2 || height < 2) return;
    const right = column + width - 1;
    const bottom = row + height - 1;
    this.set(column, row, "┌", style);
    this.set(right, row, "┐", style);
    this.set(column, bottom, "└", style);
    this.set(right, bottom, "┘", style);
    for (let x = column + 1; x < right; x++) {
      this.set(x, row, "─", style);
      this.set(x, bottom, "─", style);
    }
    for (let y = row + 1; y < bottom; y++) {
      this.set(column, y, "│", style);
      this.set(right, y, "│", style);
    }
  }

  /** Freezes the grid into a frame. */
  frame(): Frame {
    return this.#cells.map((row) => [...row]);
  }

  /** Renders to plain text -- for tests and for debugging without a terminal. */
  toText(): string {
    return this.#cells.map((row) => row.map((cell) => cell.char).join("").trimEnd()).join("\n");
  }
}
