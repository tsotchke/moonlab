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
import { MoonLabApp } from "./src/app/console_app.ts";
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
  runShellApp(presenter, new MoonLabApp({ backend }));

  /**
   * Keys reach the application only once the host's keyboard target has
   * focus. On a page whose whole content is the console, a visitor should not
   * have to click before typing works -- but exotui 0.6.0 exposes no `focus()`
   * on the host, and the target is a hidden textarea the browser platform
   * creates for on-screen keyboard support. So: focus it if we can find it,
   * fall back to the mount, and re-focus on pointerdown, which is what a user
   * does anyway.
   */
  const focusKeyboard = () => {
    const target = document.querySelector<HTMLElement>("body > textarea") ?? root;
    target.focus?.();
  };
  focusKeyboard();
  root.addEventListener("pointerdown", focusKeyboard);
} catch (cause) {
  const message = cause instanceof Error ? cause.message : String(cause);
  say(`could not start: ${message}`);
  throw cause;
}
