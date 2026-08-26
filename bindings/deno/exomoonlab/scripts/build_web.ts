#!/usr/bin/env -S deno run -A
/**
 * Builds the browser host into `dist/`.
 *
 * Three things have to land side by side: the bundled application, the
 * Emscripten glue, and the .wasm the glue fetches. The backend resolves the
 * glue as `./moonlab.js` relative to the page, so their being siblings is not
 * a convenience -- it is the contract.
 */

const here = new URL("..", import.meta.url).pathname;
const dist = `${here}dist`;

/** The newest Emscripten build available, preferring a local pnpm build. */
function findArtifacts(): { glue: string; wasm: string } {
  const repoRoot = `${here}../../../`;
  const candidates = [
    `${repoRoot}bindings/javascript/packages/core/dist`,
    `${repoRoot}bindings/javascript/demo/public`,
    `${repoRoot}docs`,
  ];
  for (const dir of candidates) {
    try {
      Deno.statSync(`${dir}/moonlab.js`);
      Deno.statSync(`${dir}/moonlab.wasm`);
      return { glue: `${dir}/moonlab.js`, wasm: `${dir}/moonlab.wasm` };
    } catch {
      // try the next one
    }
  }
  throw new Error(
    "no moonlab.js + moonlab.wasm pair found. Build one with:\n" +
      "  cd bindings/javascript/packages/core && pnpm build:wasm   (needs emcc)",
  );
}

async function run(args: string[]): Promise<void> {
  const command = new Deno.Command(Deno.execPath(), { args, stdout: "inherit", stderr: "inherit" });
  const { success } = await command.output();
  if (!success) throw new Error(`failed: deno ${args.join(" ")}`);
}

await Deno.mkdir(dist, { recursive: true });

console.log("bundling web.ts …");
// Until exotui 0.7.0 publishes, ./shell resolves only through the local
// override; use it when it is there and fall back to the pin when it is not.
const mapArgs = (() => {
  for (const candidate of [`${here}import_map.local.json`, `${here}import_map.dev.json`]) {
    try {
      Deno.statSync(candidate);
      console.log(`using exotui override: ${candidate}`);
      return [`--import-map=${candidate}`];
    } catch {
      // try the next one
    }
  }
  return [];
})();
await run(["bundle", ...mapArgs, "--platform=browser", "-o", `${dist}/exomoonlab.js`, `${here}web.ts`]);

const { glue, wasm } = findArtifacts();
console.log(`artifacts: ${glue}`);
await Deno.copyFile(glue, `${dist}/moonlab.js`);
await Deno.copyFile(wasm, `${dist}/moonlab.wasm`);
await Deno.copyFile(`${here}web/index.html`, `${dist}/index.html`);

const sizes = await Promise.all(
  ["exomoonlab.js", "moonlab.js", "moonlab.wasm", "index.html"].map(async (name) => {
    const { size } = await Deno.stat(`${dist}/${name}`);
    return `  ${name.padEnd(16)} ${(size / 1024).toFixed(0)} KB`;
  }),
);
console.log(`built into ${dist}:\n${sizes.join("\n")}`);
