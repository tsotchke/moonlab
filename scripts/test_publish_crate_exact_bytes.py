#!/usr/bin/env python3
"""End-to-end test for the exact-byte crate publisher.

The test:

  1. Packages moonlab-sys, moonlab, and moonlab-tui with
     `cargo package --no-verify`, in dependency order.  moonlab and moonlab-tui
     depend on siblings that are not yet on crates.io, so their unpublished
     path+version dependencies are made resolvable with a throwaway
     `--config patch.crates-io.<sibling>.path=...` -- this never reaches the
     packaged manifest.
  2. For each .crate: records the SHA-256 of the bytes on disk, runs the
     publisher in --dry-run with --emit-body, and asserts the emitted request
     body is exactly `u32_le(len(json)) json u32_le(len(crate)) crate` with the
     trailing segment byte-identical to the .crate on disk (SHA-256 included).
  3. Asserts the publisher fails closed when the expected SHA-256 is wrong and
     writes no body.
  4. Uploads the real bytes to an in-process HTTP registry that mimics
     `PUT /api/v1/crates/new`, and asserts the server received the exact framed
     body -- the trailing tarball matching the .crate byte for byte -- with the
     documented headers and token.

Run:  python3 scripts/test_publish_crate_exact_bytes.py
Exit: 0 on success, non-zero on the first failed assertion.
"""

from __future__ import annotations

import hashlib
import http.server
import io
import json
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_ROOT = REPO_ROOT / "bindings" / "rust"
PUBLISHER = REPO_ROOT / "scripts" / "publish_crate_exact_bytes.py"

# (crate directory name, sibling crates whose path must be patched in to resolve)
CRATES = [
    ("moonlab-sys", []),
    ("moonlab", ["moonlab-sys"]),
    ("moonlab-tui", ["moonlab-sys", "moonlab"]),
]


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def crate_version(crate_dir: Path) -> str:
    manifest = tomllib.loads((crate_dir / "Cargo.toml").read_text(encoding="utf-8"))
    return manifest["package"]["version"]


def package_crate(crate_dir: Path, siblings: list[str]) -> Path:
    cmd = ["cargo", "package", "--no-verify", "--allow-dirty"]
    for sibling in siblings:
        sibling_path = (RUST_ROOT / sibling).resolve()
        cmd += ["--config", f'patch.crates-io.{sibling}.path="{sibling_path}"']
    print(f"  packaging {crate_dir.name}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=crate_dir, capture_output=True, text=True)
    if result.returncode != 0:
        fail(f"cargo package failed for {crate_dir.name}:\n{result.stderr}")
    version = crate_version(crate_dir)
    crate_path = crate_dir / "target" / "package" / f"{crate_dir.name}-{version}.crate"
    if not crate_path.is_file():
        fail(f"expected packaged crate not found: {crate_path}")
    return crate_path


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_body(body: bytes) -> tuple[dict, bytes]:
    """Reverse the wire framing: return (metadata, crate_bytes)."""
    if len(body) < 8:
        fail("emitted body is too short to contain the framing")
    (json_len,) = struct.unpack_from("<I", body, 0)
    json_start = 4
    json_end = json_start + json_len
    metadata = json.loads(body[json_start:json_end].decode("utf-8"))
    (crate_len,) = struct.unpack_from("<I", body, json_end)
    crate_start = json_end + 4
    crate_end = crate_start + crate_len
    if crate_end != len(body):
        fail(f"framing mismatch: crate segment ends at {crate_end}, body is {len(body)} bytes")
    return metadata, body[crate_start:crate_end]


def run_publisher(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PUBLISHER), *args], capture_output=True, text=True
    )


def check_dry_run_exact_bytes(crate_path: Path, expected_sha: str, workdir: Path) -> None:
    body_path = workdir / f"{crate_path.stem}.body"
    result = run_publisher(
        [
            "publish",
            "--crate", str(crate_path),
            "--sha256", expected_sha,
            "--registry-url", "http://127.0.0.1:1/never-contacted",
            "--dry-run",
            "--emit-body", str(body_path),
        ]
    )
    if result.returncode != 0:
        fail(f"dry-run publish failed for {crate_path.name}:\n{result.stderr}")

    crate_bytes = crate_path.read_bytes()
    body = body_path.read_bytes()
    metadata, framed_crate = parse_body(body)

    if framed_crate != crate_bytes:
        fail(f"{crate_path.name}: framed crate bytes differ from the file on disk")
    if sha256_hex(framed_crate) != expected_sha:
        fail(f"{crate_path.name}: framed crate SHA-256 does not match the certified hash")

    version = crate_version(crate_path.parents[2])
    if metadata["name"] != crate_path.parents[2].name or metadata["vers"] != version:
        fail(f"{crate_path.name}: metadata name/vers mismatch: {metadata['name']} {metadata['vers']}")

    # The trailing bytes of the body are the crate verbatim; nothing re-packaged.
    if not body.endswith(crate_bytes):
        fail(f"{crate_path.name}: request body does not end with the exact .crate bytes")
    print(f"  ok: {crate_path.name} exact bytes preserved ({len(crate_bytes)} bytes, sha {expected_sha[:12]}...)")


def check_fail_closed(crate_path: Path, workdir: Path) -> None:
    wrong_sha = "0" * 64
    body_path = workdir / f"{crate_path.stem}.should-not-exist.body"
    result = run_publisher(
        [
            "publish",
            "--crate", str(crate_path),
            "--sha256", wrong_sha,
            "--registry-url", "http://127.0.0.1:1/never-contacted",
            "--dry-run",
            "--emit-body", str(body_path),
        ]
    )
    if result.returncode == 0:
        fail(f"{crate_path.name}: publisher accepted a wrong SHA-256 instead of failing closed")
    if "mismatch" not in (result.stderr + result.stdout).lower():
        fail(f"{crate_path.name}: expected a hash-mismatch error, got:\n{result.stderr}")
    if body_path.exists():
        fail(f"{crate_path.name}: a body was written despite the hash mismatch")
    print(f"  ok: {crate_path.name} fails closed on a wrong SHA-256")


class _RecordingHandler(http.server.BaseHTTPRequestHandler):
    received: dict = {}

    def do_PUT(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        _RecordingHandler.received = {
            "path": self.path,
            "body": body,
            "content_type": self.headers.get("Content-Type"),
            "accept": self.headers.get("Accept"),
            "authorization": self.headers.get("Authorization"),
            "user_agent": self.headers.get("User-Agent"),
        }
        payload = json.dumps(
            {"warnings": {"invalid_categories": [], "invalid_badges": [], "other": []}}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # silence the default stderr logging
        pass


def check_end_to_end_upload(crate_path: Path, expected_sha: str) -> None:
    server = http.server.HTTPServer(("127.0.0.1", 0), _RecordingHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    token = "test-registry-token-abc123"
    result = run_publisher(
        [
            "publish",
            "--crate", str(crate_path),
            "--sha256", expected_sha,
            "--registry-url", f"http://127.0.0.1:{port}",
            "--token", token,
            "--timeout", "30",
        ]
    )
    thread.join(timeout=10)
    server.server_close()

    if result.returncode != 0:
        fail(f"end-to-end upload failed for {crate_path.name}:\n{result.stderr}")

    received = _RecordingHandler.received
    if received.get("path") != "/api/v1/crates/new":
        fail(f"server saw wrong path: {received.get('path')}")
    if received.get("content_type") != "application/octet-stream":
        fail(f"server saw wrong Content-Type: {received.get('content_type')}")
    if received.get("accept") != "application/json":
        fail(f"server saw wrong Accept: {received.get('accept')}")
    if received.get("authorization") != token:
        fail("server saw wrong Authorization header (token must be sent verbatim)")
    if not received.get("user_agent"):
        fail("server saw no User-Agent (crates.io rejects an empty User-Agent)")

    _, framed_crate = parse_body(received["body"])
    crate_bytes = crate_path.read_bytes()
    if framed_crate != crate_bytes:
        fail(f"{crate_path.name}: bytes received by the registry differ from the .crate on disk")
    if sha256_hex(framed_crate) != expected_sha:
        fail(f"{crate_path.name}: registry received bytes whose SHA-256 is not the certified hash")
    print(f"  ok: {crate_path.name} exact bytes delivered over the wire to a local registry")


def main() -> int:
    print("Packaging crates in dependency order (cargo package --no-verify):")
    packaged: list[tuple[Path, str]] = []
    for crate_name, siblings in CRATES:
        crate_dir = RUST_ROOT / crate_name
        crate_path = package_crate(crate_dir, siblings)
        expected_sha = sha256_hex(crate_path.read_bytes())
        packaged.append((crate_path, expected_sha))

    with tempfile.TemporaryDirectory(prefix="moonlab-exact-byte-") as tmp:
        workdir = Path(tmp)
        print("\nDry-run exact-byte assertions:")
        for crate_path, expected_sha in packaged:
            check_dry_run_exact_bytes(crate_path, expected_sha, workdir)

        print("\nFail-closed assertions:")
        for crate_path, _ in packaged:
            check_fail_closed(crate_path, workdir)

    print("\nEnd-to-end upload to an in-process registry:")
    for crate_path, expected_sha in packaged:
        check_end_to_end_upload(crate_path, expected_sha)

    print("\nAll exact-byte publication checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
