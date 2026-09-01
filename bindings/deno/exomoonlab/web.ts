/**
 * The browser host.
 *
 * The counterpart to `main.ts`, and deliberately the same shape: build a
 * presenter, build the application, hand them to the loop. `MoonLabApp` is
 * byte-for-byte the object the terminal runs -- the difference between the two
 * hosts is this file and nothing else, which is the claim the presenter seam
 * exists to make good on.
 *
 * Only the backend differs, and not by choice: a browser has no FFI, so the
 * WASM build is the only way in. It is the same C surface either way.
 */

import { runShellApp, webPresenter } from "@ubernaut/exotui/web";
import { MoonLabDesktop } from "./src/app/desktop.ts";
import { openWasmBackend } from "./src/backend/mod.ts";

const root = document.querySelector<HTMLElement>("#app");
if (!root) throw new Error("Missing #app mount element.");

const status = document.querySelector<HTMLElement>("#status");
const say = (text: string) => {
  if (status) status.textContent = text;
};

say("loading the MoonLab WASM build…");

try {
  const backend = await openWasmBackend();
  say(backend.description);

  const presenter = webPresenter({ root });
  // runShellApp calls init(presenter) itself; ShellPresenter satisfies the
  // desktop's DesktopHost structurally, so it loads its persisted state from
  // IndexedDB here and from a file under the console host.
  runShellApp(presenter, new MoonLabDesktop({ backend }));

  // Keys reach the application only once the host's keyboard target has focus,
  // and on a page whose whole content is the console a visitor should not have
  // to click first. A pointer press focuses it anyway; this covers the visitor
  // who just starts typing.
  presenter.focus();
} catch (cause) {
  const message = cause instanceof Error ? cause.message : String(cause);
  say(`could not start: ${message}`);
  throw cause;
}
