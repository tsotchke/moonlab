#!/usr/bin/env python3
"""Network-free tests for scripts/publish_crate_exact.py.

These tests build tiny synthetic .crate tarballs in a temporary directory and
exercise the exact-byte publisher end to end: body framing, the SHA-256 gate,
name/version agreement, dry-run isolation from the network, a real --execute
against a loopback HTTP stub, and env-only token handling.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import socket
import struct
import tarfile
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "publish_crate_exact.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("publish_crate_exact", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


publisher = _load_module()


def _cargo_toml(name: str, vers: str) -> str:
    return (
        "[package]\n"
        f'name = "{name}"\n'
        f'version = "{vers}"\n'
        'edition = "2021"\n'
        'description = "synthetic test crate"\n'
        'license = "MIT"\n'
    )


def build_crate(
    directory: Path,
    name: str,
    vers: str,
    cargo_toml_text: str | None = None,
) -> Path:
    """Write a minimal <name>-<vers>.crate (gzipped tar) and return its path."""
    if cargo_toml_text is None:
        cargo_toml_text = _cargo_toml(name, vers)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        payload = cargo_toml_text.encode("utf-8")
        info = tarfile.TarInfo(f"{name}-{vers}/Cargo.toml")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
        src = b"// synthetic\n"
        src_info = tarfile.TarInfo(f"{name}-{vers}/src/lib.rs")
        src_info.size = len(src)
        tar.addfile(src_info, io.BytesIO(src))
    crate_path = directory / f"{name}-{vers}.crate"
    crate_path.write_bytes(buffer.getvalue())
    return crate_path


def write_metadata(
    directory: Path,
    name: str,
    vers: str,
    **overrides: Any,
) -> Path:
    document: dict[str, Any] = {
        "name": name,
        "vers": vers,
        "deps": [],
        "features": {},
        "authors": ["Moonlab Team"],
        "description": "synthetic test crate",
        "license": "MIT",
    }
    document.update(overrides)
    path = directory / f"{name}-{vers}.metadata.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class BaseCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.name = "moonlab-sys"
        self.vers = "1.2.0"
        self.crate = build_crate(self.tmp, self.name, self.vers)
        self.crate_bytes = self.crate.read_bytes()
        self.sha = publisher.sha256_hex(self.crate_bytes)
        self.metadata = write_metadata(self.tmp, self.name, self.vers)
        self.metadata_bytes = self.metadata.read_bytes()


class BodyFramingTests(BaseCase):
    def test_assemble_body_layout(self) -> None:
        body = publisher.assemble_body(self.metadata_bytes, self.crate_bytes)
        meta_len = struct.unpack("<I", body[0:4])[0]
        self.assertEqual(meta_len, len(self.metadata_bytes))
        offset = 4
        self.assertEqual(body[offset : offset + meta_len], self.metadata_bytes)
        offset += meta_len
        crate_len = struct.unpack("<I", body[offset : offset + 4])[0]
        self.assertEqual(crate_len, len(self.crate_bytes))
        offset += 4
        self.assertEqual(body[offset : offset + crate_len], self.crate_bytes)
        self.assertEqual(offset + crate_len, len(body))

    def test_dry_run_writes_body_out_with_exact_framing(self) -> None:
        body_out = self.tmp / "body.bin"
        rc = publisher.main(
            [
                "--crate-file",
                str(self.crate),
                "--index-metadata",
                str(self.metadata),
                "--expected-sha256",
                self.sha,
                "--body-out",
                str(body_out),
            ]
        )
        self.assertEqual(rc, 0)
        self.assertTrue(body_out.exists())
        body = body_out.read_bytes()
        expected = publisher.assemble_body(self.metadata_bytes, self.crate_bytes)
        self.assertEqual(body, expected)
        # The 4-byte LE length prefixes must sit at the exact offsets.
        self.assertEqual(struct.unpack("<I", body[0:4])[0], len(self.metadata_bytes))
        crate_prefix = 4 + len(self.metadata_bytes)
        self.assertEqual(
            struct.unpack("<I", body[crate_prefix : crate_prefix + 4])[0],
            len(self.crate_bytes),
        )


class Sha256GateTests(BaseCase):
    def test_refuses_on_hash_mismatch(self) -> None:
        wrong = "0" * 64
        body_out = self.tmp / "body.bin"
        rc = publisher.main(
            [
                "--crate-file",
                str(self.crate),
                "--index-metadata",
                str(self.metadata),
                "--expected-sha256",
                wrong,
                "--body-out",
                str(body_out),
            ]
        )
        self.assertEqual(rc, 2)
        # A refused upload must not have assembled or written a body.
        self.assertFalse(body_out.exists())

    def test_refuses_malformed_hash(self) -> None:
        rc = publisher.main(
            [
                "--crate-file",
                str(self.crate),
                "--index-metadata",
                str(self.metadata),
                "--expected-sha256",
                "deadbeef",
            ]
        )
        self.assertEqual(rc, 2)


class NameVersionAgreementTests(BaseCase):
    def test_filename_mismatch_refused(self) -> None:
        # Metadata claims a version the crate filename does not carry.
        bad_meta = write_metadata(self.tmp, self.name, "9.9.9")
        rc = publisher.main(
            [
                "--crate-file",
                str(self.crate),
                "--index-metadata",
                str(bad_meta),
                "--expected-sha256",
                self.sha,
            ]
        )
        self.assertEqual(rc, 2)

    def test_embedded_cargo_toml_mismatch_refused(self) -> None:
        # Filename and metadata agree, but the embedded Cargo.toml disagrees.
        crate = build_crate(
            self.tmp,
            "moonlab",
            "1.2.0",
            cargo_toml_text=_cargo_toml("moonlab", "0.0.1"),
        )
        meta = write_metadata(self.tmp, "moonlab", "1.2.0")
        rc = publisher.main(
            [
                "--crate-file",
                str(crate),
                "--index-metadata",
                str(meta),
                "--expected-sha256",
                publisher.sha256_hex(crate.read_bytes()),
            ]
        )
        self.assertEqual(rc, 2)


class NoNetworkOnDryRunTests(BaseCase):
    def test_dry_run_never_touches_the_network(self) -> None:
        original_socket = socket.socket
        original_urlopen = publisher.urllib.request.urlopen

        def forbidden_socket(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("dry-run opened a socket")

        def forbidden_urlopen(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("dry-run called urlopen")

        socket.socket = forbidden_socket  # type: ignore[assignment]
        publisher.urllib.request.urlopen = forbidden_urlopen  # type: ignore[assignment]
        try:
            rc = publisher.main(
                [
                    "--crate-file",
                    str(self.crate),
                    "--index-metadata",
                    str(self.metadata),
                    "--expected-sha256",
                    self.sha,
                    "--dry-run",
                ]
            )
        finally:
            socket.socket = original_socket  # type: ignore[assignment]
            publisher.urllib.request.urlopen = original_urlopen  # type: ignore[assignment]
        self.assertEqual(rc, 0)


class _RecordingHandler(BaseHTTPRequestHandler):
    def do_PUT(self) -> None:  # noqa: N802 - required handler name
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.server.recorded = {  # type: ignore[attr-defined]
            "path": self.path,
            "authorization": self.headers.get("Authorization"),
            "content_type": self.headers.get("Content-Type"),
            "body": body,
        }
        payload = json.dumps({"warnings": {}}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:  # silence test output
        return


class ExecuteTests(BaseCase):
    def test_execute_uploads_exact_body_and_reads_token_from_env(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), _RecordingHandler)
        server.recorded = None  # type: ignore[attr-defined]
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            registry = f"http://{host}:{port}"
            token = "test-token-value-never-logged"
            env_name = "MOONLAB_TEST_REGISTRY_TOKEN"
            os.environ[env_name] = token
            try:
                rc = publisher.main(
                    [
                        "--crate-file",
                        str(self.crate),
                        "--index-metadata",
                        str(self.metadata),
                        "--registry-url",
                        registry,
                        "--token-env",
                        env_name,
                        "--expected-sha256",
                        self.sha,
                        "--execute",
                    ]
                )
            finally:
                os.environ.pop(env_name, None)
        finally:
            thread.join(timeout=5)
            server.server_close()

        self.assertEqual(rc, 0)
        recorded = server.recorded  # type: ignore[attr-defined]
        self.assertIsNotNone(recorded)
        self.assertEqual(recorded["path"], "/api/v1/crates/new")
        self.assertEqual(recorded["authorization"], token)
        expected_body = publisher.assemble_body(self.metadata_bytes, self.crate_bytes)
        self.assertEqual(recorded["body"], expected_body)

    def test_execute_requires_token_env_set(self) -> None:
        env_name = "MOONLAB_TEST_MISSING_TOKEN"
        os.environ.pop(env_name, None)
        # No server is started; a missing token must fail before any request.
        rc = publisher.main(
            [
                "--crate-file",
                str(self.crate),
                "--index-metadata",
                str(self.metadata),
                "--registry-url",
                "http://127.0.0.1:1",
                "--token-env",
                env_name,
                "--expected-sha256",
                self.sha,
                "--execute",
            ]
        )
        self.assertEqual(rc, 2)

    def test_no_token_argument_is_accepted(self) -> None:
        # The token must never be passable on argv.
        with self.assertRaises(SystemExit):
            publisher.parse_args(
                [
                    "--crate-file",
                    str(self.crate),
                    "--index-metadata",
                    str(self.metadata),
                    "--expected-sha256",
                    self.sha,
                    "--token",
                    "secret",
                ]
            )


class ResponseHandlingTests(BaseCase):
    def test_non_2xx_is_hard_failure(self) -> None:
        body = json.dumps({"errors": [{"detail": "crate exists"}]}).encode("utf-8")
        with self.assertRaises(publisher.PublishError) as ctx:
            publisher.interpret_response(403, body)
        self.assertIn("HTTP 403", str(ctx.exception))
        self.assertIn("crate exists", str(ctx.exception))

    def test_errors_in_2xx_body_still_fail(self) -> None:
        body = json.dumps({"errors": [{"detail": "bad metadata"}]}).encode("utf-8")
        with self.assertRaises(publisher.PublishError):
            publisher.interpret_response(200, body)

    def test_warnings_do_not_fail(self) -> None:
        body = json.dumps(
            {"warnings": {"other": ["heads up"], "invalid_categories": []}}
        ).encode("utf-8")
        publisher.interpret_response(200, body)


if __name__ == "__main__":
    unittest.main()
