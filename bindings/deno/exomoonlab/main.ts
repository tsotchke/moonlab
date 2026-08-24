#!/usr/bin/env -S deno run --allow-read --allow-ffi --allow-env --allow-write
/**
 * The terminal host.
 *
 * `runConsoleShellApp` is exotui's console-side loop: it builds the console
 * presenter, subscribes the application to input, and drives frames. The
 * application itself knows none of that -- swapping this file for the browser
 * host is the whole of "running it in a browser".
 */

import { runConsoleShellApp } from "@ubernaut/exotui/runtime";
import { MoonLabApp } from "./src/app/console_app.ts";
import { type BackendKind, selectBackend } from "./src/backend/mod.ts";

function parseBackend(args: string[]): BackendKind | undefined {
  const flag = args.find((a) => a.startsWith("--backend="));
  const value = flag?.split("=")[1];
  if (value === "native" || value === "wasm") return value;
  if (value !== undefined) throw new Error(`--backend must be native or wasm, got ${value}`);
  return undefined;
}

if (import.meta.main) {
  const only = parseBackend(Deno.args);
  const backend = await selectBackend(only ? { only } : {});

  // Printed before entering the alternate screen, so it survives the exit.
  console.error(`exomoonlab: ${backend.description}`);

  // The app must be able to stop the loop, and the loop needs the app to
  // exist first. A const holder carries the reference across that cycle.
  const loop: { handle?: ReturnType<typeof runConsoleShellApp> } = {};
  const app = new MoonLabApp({
    backend,
    onQuit: () => {
      loop.handle?.stop();
      backend.dispose().finally(() => Deno.exit(0));
    },
  });

  loop.handle = runConsoleShellApp(app);
}
