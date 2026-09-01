import type {
  KeyPressEvent,
  MouseScrollEvent,
  PointerInputEvent,
  Rectangle,
  ShellApp,
  ShellBorderGlyphs,
  ShellPresentedCell,
  ShellPresentedFrame,
  ShellPresenter,
  ShellRgb,
  ShellStyle,
  ShellSurface,
} from "@ubernaut/exotui/shell";

// Does ./shell name the store type at all?
type StoreFromShell = ReturnType<ShellPresenter["store"]>;
const _s: StoreFromShell | undefined = undefined;

const _c: ShellPresentedCell = { char: "x" };
const _f: ShellPresentedFrame = [[_c]];
const _r: Rectangle = { column: 0, row: 0, width: 1, height: 1 };
const _st: ShellStyle = { foreground: [1, 2, 3] as ShellRgb };
declare const _su: ShellSurface;
declare const _bg: ShellBorderGlyphs;
declare const _k: KeyPressEvent;
declare const _p: PointerInputEvent;
declare const _w: MouseScrollEvent;
declare const _a: ShellApp;
console.log(_s, _f, _r, _st, _su, _bg, _k, _p, _w, _a);
