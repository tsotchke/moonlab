#!/usr/bin/env python3
"""Publish a pre-built, content-certified .crate to a Cargo registry byte-for-byte.

The v1.2.0 release pipeline builds each crate (moonlab-sys, moonlab, moonlab-tui)
in a hosted candidate run and content-hashes the resulting ``.crate`` tarball into
a release certificate.  ``cargo publish`` cannot promote those artifacts because it
re-packages the crate from source, producing bytes that no longer match the
certificate.  This tool uploads the certified ``.crate`` verbatim.

The registry publish endpoint is ``PUT {registry}/api/v1/crates/new`` with a framed
body identical to the one cargo assembles:

    <u32 le: length of metadata JSON><metadata JSON bytes>
    <u32 le: length of crate bytes><crate tarball bytes>

Authentication is the registry token, sent verbatim in the ``Authorization`` header.
The token is only ever read from an environment variable and is never accepted on
the command line, printed, or logged.

The default mode is a dry run: the exact body is assembled and summarized (and
optionally written out) but nothing is sent.  Uploading requires ``--execute``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import struct
import sys
import tarfile
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY_URL = "https://crates.io"
DEFAULT_TOKEN_ENV = "CARGO_REGISTRY_TOKEN"
PUBLISH_PATH = "/api/v1/crates/new"
USER_AGENT = "moonlab-publish-crate-exact/1.2.0"
REQUEST_TIMEOUT_SECONDS = 120

# Framed length prefixes are 4-byte little-endian unsigned integers, matching
# cargo's registry protocol (see cargo src/cargo/ops/registry/publish.rs).
LENGTH_PREFIX = struct.Struct("<I")
MAX_FRAME_BYTES = 0xFFFF_FFFF


class PublishError(RuntimeError):
    """A fail-closed error that aborts publication.

    The message must never contain the registry token or any request header.
    """


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_metadata(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Read the publish metadata document, returning its exact bytes and parsed form.

    The raw bytes are embedded in the upload verbatim so the certified document is
    never mutated; the parsed dictionary is used only for validation.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PublishError(f"cannot read metadata file {path}: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PublishError(f"metadata file {path} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise PublishError(f"metadata file {path} must contain a JSON object")
    for field in ("name", "vers"):
        value = parsed.get(field)
        if not isinstance(value, str) or not value:
            raise PublishError(
                f"metadata file {path} is missing a non-empty '{field}' string"
            )
    return raw, parsed


def read_crate_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise PublishError(f"cannot read crate file {path}: {exc}") from exc


def check_filename(crate_path: Path, name: str, vers: str) -> None:
    expected = f"{name}-{vers}.crate"
    if crate_path.name != expected:
        raise PublishError(
            f"crate filename {crate_path.name!r} does not match metadata "
            f"name/version (expected {expected!r})"
        )


def check_sha256(crate_bytes: bytes, expected_sha256: str) -> str:
    expected = expected_sha256.strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise PublishError(
            "--expected-sha256 must be a 64-character hex SHA-256 digest"
        )
    actual = sha256_hex(crate_bytes)
    if actual != expected:
        raise PublishError(
            "crate content hash does not match the certificate; refusing to upload "
            f"(expected {expected}, got {actual})"
        )
    return actual


def check_embedded_cargo_toml(crate_bytes: bytes, name: str, vers: str) -> None:
    """Validate the tarball's embedded Cargo.toml agrees on name and version.

    A published crate stores its files under ``<name>-<vers>/``; the generated
    manifest at ``<name>-<vers>/Cargo.toml`` carries concrete name and version
    strings that must match the metadata document.
    """
    member = f"{name}-{vers}/Cargo.toml"
    try:
        with tarfile.open(fileobj=io.BytesIO(crate_bytes), mode="r:gz") as tar:
            try:
                info = tar.getmember(member)
            except KeyError as exc:
                raise PublishError(
                    f"crate tarball does not contain expected manifest {member!r}"
                ) from exc
            extracted = tar.extractfile(info)
            if extracted is None:
                raise PublishError(f"manifest {member!r} is not a regular file")
            manifest_bytes = extracted.read()
    except tarfile.TarError as exc:
        raise PublishError(f"crate file is not a readable tar.gz archive: {exc}") from exc

    try:
        manifest = tomllib.loads(manifest_bytes.decode("utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise PublishError(f"embedded {member!r} is not valid TOML: {exc}") from exc

    package = manifest.get("package")
    if not isinstance(package, dict):
        raise PublishError(f"embedded {member!r} has no [package] table")
    embedded_name = package.get("name")
    embedded_vers = package.get("version")
    if embedded_name != name:
        raise PublishError(
            f"embedded manifest name {embedded_name!r} does not match metadata "
            f"name {name!r}"
        )
    if embedded_vers != vers:
        raise PublishError(
            f"embedded manifest version {embedded_vers!r} does not match metadata "
            f"version {vers!r}"
        )


def assemble_body(metadata_bytes: bytes, crate_bytes: bytes) -> bytes:
    """Frame the metadata and crate bytes into the registry upload body.

    Layout: 4-byte LE metadata length, metadata bytes, 4-byte LE crate length,
    crate bytes.
    """
    if len(metadata_bytes) > MAX_FRAME_BYTES:
        raise PublishError("metadata document exceeds the 4 GiB frame limit")
    if len(crate_bytes) > MAX_FRAME_BYTES:
        raise PublishError("crate file exceeds the 4 GiB frame limit")
    return b"".join(
        (
            LENGTH_PREFIX.pack(len(metadata_bytes)),
            metadata_bytes,
            LENGTH_PREFIX.pack(len(crate_bytes)),
            crate_bytes,
        )
    )


def publish_url(registry_url: str) -> str:
    return registry_url.rstrip("/") + PUBLISH_PATH


def read_token(token_env: str) -> str:
    token = os.environ.get(token_env)
    if not token:
        raise PublishError(
            f"registry token environment variable {token_env!r} is not set"
        )
    return token


def send_body(url: str, token: str, body: bytes) -> tuple[int, bytes]:
    """PUT the framed body and return (status, response_bytes).

    The token is placed in the Authorization header only; it is never returned,
    printed, or included in any error surfaced by this function.
    """
    request = urllib.request.Request(
        url,
        data=body,
        method="PUT",
        headers={
            "Authorization": token,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            "Content-Length": str(len(body)),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        # Read the error body so a non-2xx response can be reported; the body is
        # registry JSON and does not echo the request token.
        detail = exc.read()
        return exc.code, detail
    except urllib.error.URLError as exc:
        raise PublishError(f"registry request failed: {exc.reason}") from exc


def interpret_response(status: int, body: bytes) -> None:
    """Raise on failure; print warnings on success. Never echoes the token."""
    text = body.decode("utf-8", errors="replace")
    parsed: Any = None
    try:
        parsed = json.loads(text) if text.strip() else None
    except json.JSONDecodeError:
        parsed = None

    errors: list[str] = []
    if isinstance(parsed, dict):
        raw_errors = parsed.get("errors")
        if isinstance(raw_errors, list):
            for item in raw_errors:
                if isinstance(item, dict) and "detail" in item:
                    errors.append(str(item["detail"]))
                else:
                    errors.append(str(item))

    if status < 200 or status >= 300:
        summary = "; ".join(errors) if errors else text.strip() or "(empty body)"
        raise PublishError(f"registry returned HTTP {status}: {summary}")

    if errors:
        raise PublishError(
            "registry accepted the request but reported errors: " + "; ".join(errors)
        )

    if isinstance(parsed, dict):
        warnings = parsed.get("warnings")
        if isinstance(warnings, dict):
            for key in ("invalid_categories", "invalid_badges", "other"):
                for message in warnings.get(key) or []:
                    print(f"warning ({key}): {message}", file=sys.stderr)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="publish_crate_exact.py",
        description="Upload a certified .crate to a Cargo registry byte-for-byte.",
        # Disable prefix abbreviation so a stray --token cannot be silently
        # matched to --token-env and mistaken for the token value.
        allow_abbrev=False,
    )
    parser.add_argument(
        "--crate-file",
        type=Path,
        required=True,
        help="Path to the certified .crate tarball to upload.",
    )
    parser.add_argument(
        "--index-metadata",
        type=Path,
        required=True,
        help="Path to the publish metadata JSON document (name, vers, deps, ...).",
    )
    parser.add_argument(
        "--registry-url",
        default=DEFAULT_REGISTRY_URL,
        help=f"Registry base URL (default: {DEFAULT_REGISTRY_URL}).",
    )
    parser.add_argument(
        "--token-env",
        default=DEFAULT_TOKEN_ENV,
        help=(
            "Name of the environment variable holding the registry token "
            f"(default: {DEFAULT_TOKEN_ENV}). The token is never taken on argv."
        ),
    )
    parser.add_argument(
        "--expected-sha256",
        required=True,
        help="Certified SHA-256 of the .crate; upload is refused on mismatch.",
    )
    parser.add_argument(
        "--body-out",
        type=Path,
        default=None,
        help="Optional path to write the assembled request body.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        dest="execute",
        action="store_false",
        help="Assemble and summarize the body without sending (default).",
    )
    mode.add_argument(
        "--execute",
        dest="execute",
        action="store_true",
        help="Actually upload the body to the registry.",
    )
    parser.set_defaults(execute=False)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    metadata_bytes, metadata = load_metadata(args.index_metadata)
    name = metadata["name"]
    vers = metadata["vers"]

    check_filename(args.crate_file, name, vers)
    crate_bytes = read_crate_bytes(args.crate_file)
    crate_sha256 = check_sha256(crate_bytes, args.expected_sha256)
    check_embedded_cargo_toml(crate_bytes, name, vers)

    body = assemble_body(metadata_bytes, crate_bytes)
    body_sha256 = sha256_hex(body)

    if args.body_out is not None:
        try:
            args.body_out.write_bytes(body)
        except OSError as exc:
            raise PublishError(f"cannot write body to {args.body_out}: {exc}") from exc

    print(f"crate: {name} {vers} ({args.crate_file.name})")
    print(f"crate sha256: {crate_sha256}")
    print(f"metadata bytes: {len(metadata_bytes)}")
    print(f"crate bytes: {len(crate_bytes)}")
    print(f"body bytes: {len(body)}")
    print(f"body sha256: {body_sha256}")

    if not args.execute:
        print("dry-run: request assembled but not sent")
        if args.body_out is not None:
            print(f"dry-run: body written to {args.body_out}")
        return 0

    token = read_token(args.token_env)
    url = publish_url(args.registry_url)
    print(f"publishing to {url}")
    status, response_body = send_body(url, token, body)
    interpret_response(status, response_body)
    print(f"published {name} {vers}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except PublishError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
