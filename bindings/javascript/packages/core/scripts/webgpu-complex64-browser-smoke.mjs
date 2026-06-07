#!/usr/bin/env node
import { spawn } from 'node:child_process';
import {
  createReadStream,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { createServer } from 'node:http';
import { dirname, extname, join, relative, resolve, sep } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';

const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = parseArgs(process.argv.slice(2));
const browserPath = args.browser || process.env.MOONLAB_BROWSER || findBrowserPath();

if (!browserPath) {
  throw new Error(
    'No Chrome-compatible browser found. Set MOONLAB_BROWSER or pass --browser <path>.'
  );
}

const server = createStaticServer(packageRoot);
const profileDir = mkdtempSync(join(tmpdir(), 'moonlab-webgpu-browser-'));

try {
  const port = await listen(server, args.port);
  const url = buildHarnessUrl(port, args);
  const artifactJson = await runBrowserHarness({
    browserPath,
    profileDir,
    url,
  });
  const artifact = JSON.parse(artifactJson);
  const output = `${artifactJson.trim()}\n`;

  if (args.out) {
    const outputPath = resolve(args.out);
    mkdirSync(dirname(outputPath), { recursive: true });
    writeFileSync(outputPath, output);
  } else {
    process.stdout.write(output);
  }

  if (
    artifact.schema === 'moonlab.webgpu.complex64-browser-harness-error.v0'
    || artifact.contractValidation?.valid !== true
    || (args.requireBackend && artifact.webgpuParity?.executed !== true)
  ) {
    process.exitCode = 1;
  }
} finally {
  await closeServer(server);
  if (!args.keepProfile) {
    rmSync(profileDir, { recursive: true, force: true });
  }
}

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--') {
      continue;
    } else if (arg === '--browser') {
      parsed.browser = argv[++index];
    } else if (arg === '--out') {
      parsed.out = argv[++index];
    } else if (arg === '--generated-at') {
      parsed.generatedAt = argv[++index];
    } else if (arg === '--port') {
      parsed.port = Number.parseInt(argv[++index], 10);
    } else if (arg === '--canonical') {
      parsed.canonical = true;
    } else if (arg === '--require-backend') {
      parsed.requireBackend = true;
    } else if (arg === '--keep-profile') {
      parsed.keepProfile = true;
    } else if (arg === '--help' || arg === '-h') {
      process.stdout.write(
        [
          'Usage: node scripts/webgpu-complex64-browser-smoke.mjs [options]',
          '',
          'Options:',
          '  --browser <path>       Chrome-compatible browser binary',
          '  --out <path>           Write extracted artifact JSON',
          '  --generated-at <iso>   Override artifact timestamp',
          '  --canonical            Emit canonical JSON from the browser harness',
          '  --require-backend      Exit nonzero unless browser WebGPU probe executed',
          '  --port <number>        Static server port, default is ephemeral',
          '  --keep-profile         Keep the temporary browser profile',
          '',
        ].join('\n')
      );
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return parsed;
}

function findBrowserPath() {
  const candidates = [
    '/bin/google-chrome',
    '/bin/google-chrome-stable',
    '/usr/bin/google-chrome',
    '/usr/bin/google-chrome-stable',
    '/bin/chromium',
    '/usr/bin/chromium',
    '/bin/chromium-browser',
    '/usr/bin/chromium-browser',
    '/bin/brave-browser',
    '/usr/bin/brave-browser',
  ];
  return candidates.find((candidate) => existsSync(candidate));
}

function createStaticServer(root) {
  return createServer((request, response) => {
    const requestUrl = new URL(request.url || '/', 'http://127.0.0.1');
    const pathname = requestUrl.pathname === '/'
      ? '/browser/webgpu-complex64-parity.html'
      : requestUrl.pathname;
    const filePath = resolve(root, `.${decodeURIComponent(pathname)}`);
    const relativePath = relative(root, filePath);

    if (relativePath.startsWith('..') || relativePath.includes(`..${sep}`)) {
      response.writeHead(403);
      response.end('Forbidden');
      return;
    }
    if (!existsSync(filePath)) {
      response.writeHead(404);
      response.end('Not found');
      return;
    }

    response.writeHead(200, {
      'Content-Type': contentType(filePath),
      'Cache-Control': 'no-store',
    });
    createReadStream(filePath).pipe(response);
  });
}

function listen(server, requestedPort) {
  return new Promise((resolveListen, reject) => {
    server.once('error', reject);
    server.listen(requestedPort || 0, '127.0.0.1', () => {
      server.off('error', reject);
      const address = server.address();
      if (!address || typeof address === 'string') {
        reject(new Error('Static server did not expose a TCP port'));
        return;
      }
      resolveListen(address.port);
    });
  });
}

function closeServer(server) {
  return new Promise((resolveClose, reject) => {
    server.close((error) => {
      if (error) {
        reject(error);
        return;
      }
      resolveClose();
    });
  });
}

function buildHarnessUrl(port, options) {
  const params = new URLSearchParams();
  if (options.generatedAt) {
    params.set('generatedAt', options.generatedAt);
  }
  if (options.canonical) {
    params.set('canonical', '1');
  }
  if (options.requireBackend) {
    params.set('requireBackend', '1');
  }
  const suffix = params.toString() ? `?${params}` : '';
  return `http://127.0.0.1:${port}/browser/webgpu-complex64-parity.html${suffix}`;
}

async function runBrowserHarness({ browserPath, profileDir, url }) {
  const browser = launchBrowser({
    browserPath,
    profileDir,
    url,
  });
  let cdp;
  try {
    const devtoolsPort = await waitForDevToolsPort(profileDir, browser);
    const pageWebSocketUrl = await waitForPageWebSocketUrl(devtoolsPort, url, browser);
    cdp = await connectCdp(pageWebSocketUrl);
    await cdp.send('Runtime.enable');
    return await waitForHarnessArtifact(cdp, browser);
  } finally {
    cdp?.close();
    browser.kill();
    await browser.closed;
  }
}

function launchBrowser({ browserPath, profileDir, url }) {
  const browserArgs = [
    '--headless=new',
    '--disable-background-networking',
    '--disable-component-update',
    '--disable-default-apps',
    '--disable-sync',
    '--no-default-browser-check',
    '--no-first-run',
    '--enable-unsafe-webgpu',
    '--remote-debugging-port=0',
    `--user-data-dir=${profileDir}`,
    url,
  ];
  if (process.getuid?.() === 0) {
    browserArgs.unshift('--no-sandbox');
  }

  const child = spawn(browserPath, browserArgs, {
    stdio: ['ignore', 'ignore', 'pipe'],
  });
  let stderr = '';
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (chunk) => {
    stderr += chunk;
  });
  child.stderrText = () => stderr.trim();
  child.closed = new Promise((resolveClosed) => {
    child.once('close', (code) => {
      resolveClosed(code);
    });
  });
  return child;
}

async function waitForDevToolsPort(profileDir, browser, timeoutMs = 15000) {
  const portFile = join(profileDir, 'DevToolsActivePort');
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (existsSync(portFile)) {
      const [port] = readFileSync(portFile, 'utf8').trim().split('\n');
      const parsedPort = Number.parseInt(port, 10);
      if (Number.isInteger(parsedPort) && parsedPort > 0) {
        return parsedPort;
      }
    }
    if (browser.exitCode !== null) {
      throw new Error(`Browser exited before DevTools was ready: ${browser.stderrText()}`);
    }
    await sleep(100);
  }
  throw new Error(`Timed out waiting for Chrome DevTools port: ${browser.stderrText()}`);
}

async function waitForPageWebSocketUrl(port, harnessUrl, browser, timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs;
  const harnessPath = new URL(harnessUrl).pathname;
  while (Date.now() < deadline) {
    if (browser.exitCode !== null) {
      throw new Error(`Browser exited before harness page was ready: ${browser.stderrText()}`);
    }
    const response = await fetch(`http://127.0.0.1:${port}/json/list`);
    const targets = await response.json();
    const target = targets.find((candidate) => (
      candidate.type === 'page'
      && typeof candidate.url === 'string'
      && new URL(candidate.url).pathname === harnessPath
      && candidate.webSocketDebuggerUrl
    )) ?? targets.find((candidate) => (
      candidate.type === 'page' && candidate.webSocketDebuggerUrl
    ));
    if (target) {
      return target.webSocketDebuggerUrl;
    }
    await sleep(100);
  }
  throw new Error(`Timed out waiting for harness page target: ${browser.stderrText()}`);
}

function connectCdp(webSocketDebuggerUrl) {
  return new Promise((resolveConnect, reject) => {
    const socket = new WebSocket(webSocketDebuggerUrl);
    const pending = new Map();
    let nextId = 1;

    socket.addEventListener('open', () => {
      resolveConnect({
        send(method, params = {}) {
          const id = nextId;
          nextId += 1;
          socket.send(JSON.stringify({ id, method, params }));
          return new Promise((resolveSend, rejectSend) => {
            pending.set(id, { resolve: resolveSend, reject: rejectSend, method });
          });
        },
        close() {
          socket.close();
        },
      });
    }, { once: true });
    socket.addEventListener('message', (event) => {
      const message = JSON.parse(String(event.data));
      if (!message.id || !pending.has(message.id)) {
        return;
      }
      const pendingCall = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) {
        pendingCall.reject(
          new Error(`${pendingCall.method} failed: ${JSON.stringify(message.error)}`)
        );
        return;
      }
      pendingCall.resolve(message.result);
    });
    socket.addEventListener('error', () => {
      reject(new Error('Chrome DevTools WebSocket failed'));
    }, { once: true });
    socket.addEventListener('close', () => {
      for (const pendingCall of pending.values()) {
        pendingCall.reject(new Error('Chrome DevTools WebSocket closed'));
      }
      pending.clear();
    });
  });
}

async function waitForHarnessArtifact(cdp, browser, timeoutMs = 20000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (browser.exitCode !== null) {
      throw new Error(`Browser exited before harness artifact was ready: ${browser.stderrText()}`);
    }
    const result = await cdp.send('Runtime.evaluate', {
      expression: `(() => {
        const output = document.getElementById('artifact-json');
        return {
          status: document.documentElement.dataset.moonlabStatus || null,
          ready: output?.dataset.ready || null,
          text: output?.textContent || ''
        };
      })()`,
      returnByValue: true,
      awaitPromise: true,
    });
    const value = result.result?.value;
    if (value?.ready === 'true' && value.text && value.text.trim() !== '{}') {
      return value.text.trim();
    }
    await sleep(200);
  }
  throw new Error(`Timed out waiting for browser harness artifact: ${browser.stderrText()}`);
}

function sleep(ms) {
  return new Promise((resolveSleep) => {
    setTimeout(resolveSleep, ms);
  });
}

function contentType(filePath) {
  if (extname(filePath) === '.html') {
    return 'text/html; charset=utf-8';
  }
  if (extname(filePath) === '.mjs' || extname(filePath) === '.js') {
    return 'text/javascript; charset=utf-8';
  }
  if (extname(filePath) === '.wasm') {
    return 'application/wasm';
  }
  if (extname(filePath) === '.json') {
    return 'application/json; charset=utf-8';
  }
  return 'application/octet-stream';
}
