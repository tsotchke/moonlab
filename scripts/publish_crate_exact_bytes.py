#!/usr/bin/env python3
"""Publish an already-built .crate to a Cargo registry using its exact bytes.

Moonlab certifies the SHA-256 of each Rust crate tarball in the release
certificate.  `cargo publish` re-runs packaging at publish time and produces a
fresh tarball, so its bytes -- and therefore its hash -- do not match the
certified artifact.  This tool never packages anything.  It reads a .crate file
that was produced earlier by `cargo package`, verifies the bytes against the
certified SHA-256, and uploads those exact bytes through the raw registry API.

Wire format (authoritative source: the Cargo book, "Web API" -> Publish, and
crates/crates-io/lib.rs in rust-lang/cargo):

    PUT {registry}/api/v1/crates/new
      Authorization: <token>                      (raw token, no scheme prefix)
      Content-Type:  application/octet-stream
      Accept:        application/json

    body =
        u32_le(len(metadata_json))   # 4 bytes, little-endian
        metadata_json                # JSON object (the crate metadata)
        u32_le(len(crate_tarball))   # 4 bytes, little-endian
        crate_tarball                # the exact .crate bytes, verbatim

The .crate bytes placed in the body are the bytes read from disk, byte for
byte.  They are never regenerated.

Subcommands
-----------
publish   Verify a .crate against its expected SHA-256 and upload it.
metadata  Emit the registry metadata JSON derived from a .crate's embedded,
          Cargo-normalized manifest (Cargo.toml + Cargo.toml.orig readme).

Authentication
--------------
The registry token is read from --token or, failing that, the
CARGO_REGISTRY_TOKEN environment variable -- the same variable `cargo publish`
uses.  crates.io wants the token verbatim in the Authorization header; it is
NOT prefixed with "Bearer" or "token".  --dry-run needs no token.

Failure modes (all fail closed -- non-zero exit, nothing uploaded)
------------------------------------------------------------------
- The .crate file is missing or unreadable.
- The computed SHA-256 does not equal the expected SHA-256.  Checked before any
  request body is built and before the network is touched.
- The length of the metadata or the tarball does not fit in a u32.
- The metadata lacks a name or version.
- No token is available for a real (non-dry-run) upload.
- The registry answers with a non-2xx status; its error detail is surfaced.

This module has no third-party dependencies -- standard library only.
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
from typing import Any

USER_AGENT = "moonlab-exact-byte-publisher/1.0"
DEFAULT_REGISTRY = "https://crates.io"
U32_MAX = 0xFFFF_FFFF


class PublishError(Exception):
    """A fatal, fail-closed condition.  Reported to stderr; exits non-zero."""


# --------------------------------------------------------------------------- #
# Reading and hashing the .crate
# --------------------------------------------------------------------------- #

def read_crate_bytes(crate_path: str) -> bytes:
    try:
        with open(crate_path, "rb") as handle:
            return handle.read()
    except OSError as exc:
        raise PublishError(f"cannot read .crate file {crate_path!r}: {exc}") from exc


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_expected_hash(expected: str) -> str:
    """Accept a bare hex digest or a `shasum`-style `<hex>  <name>` line."""
    token = expected.strip().split()[0] if expected.strip() else ""
    token = token.lower()
    if len(token) != 64 or any(character not in "0123456789abcdef" for character in token):
        raise PublishError(
            f"expected SHA-256 is not a 64-character hex digest: {expected!r}"
        )
    return token


def read_expected_hash(sha256: str | None, sha256_file: str | None) -> str:
    if sha256 and sha256_file:
        raise PublishError("pass only one of --sha256 or --sha256-file")
    if sha256:
        return normalize_expected_hash(sha256)
    if sha256_file:
        try:
            with open(sha256_file, "r", encoding="utf-8") as handle:
                return normalize_expected_hash(handle.read())
        except OSError as exc:
            raise PublishError(f"cannot read --sha256-file {sha256_file!r}: {exc}") from exc
    raise PublishError("an expected hash is required: pass --sha256 or --sha256-file")


def verify_hash(crate_bytes: bytes, expected_hex: str, crate_path: str) -> str:
    actual = sha256_hex(crate_bytes)
    if actual != expected_hex:
        raise PublishError(
            "SHA-256 mismatch -- refusing to publish.\n"
            f"  file:     {crate_path}\n"
            f"  expected: {expected_hex}\n"
            f"  actual:   {actual}\n"
            "The .crate on disk is not the certified artifact.  Re-package from "
            "the certified commit or correct the expected hash; this tool will "
            "not re-package or upload mismatched bytes."
        )
    return actual


# --------------------------------------------------------------------------- #
# Deriving registry metadata from the crate's own normalized manifest
# --------------------------------------------------------------------------- #

def _extract_member(tar: tarfile.TarFile, root: str, relpath: str) -> bytes | None:
    try:
        member = tar.getmember(f"{root}/{relpath}")
    except KeyError:
        return None
    handle = tar.extractfile(member)
    if handle is None:
        return None
    return handle.read()


def _crate_root(tar: tarfile.TarFile) -> str:
    for name in tar.getnames():
        top = name.split("/", 1)[0]
        if top:
            return top
    raise PublishError("the .crate archive is empty")


def _dep_entries(table: dict[str, Any], kind: str, target: str | None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for toml_name, spec in table.items():
        if isinstance(spec, str):
            spec = {"version": spec}
        if not isinstance(spec, dict):
            raise PublishError(f"unexpected dependency spec for {toml_name!r}")
        renamed = spec.get("package")
        crate_name = renamed if renamed else toml_name
        entry = {
            "name": crate_name,
            "version_req": str(spec.get("version", "*")),
            "features": list(spec.get("features", [])),
            "optional": bool(spec.get("optional", False)),
            "default_features": bool(spec.get("default-features", True)),
            "target": target,
            "kind": kind,
            "registry": spec.get("registry-index") or spec.get("registry"),
            "explicit_name_in_toml": toml_name if renamed else None,
        }
        entries.append(entry)
    entries.sort(key=lambda item: (item["kind"], item["name"]))
    return entries


def _collect_deps(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    deps: list[dict[str, Any]] = []
    deps += _dep_entries(manifest.get("dependencies", {}), "normal", None)
    deps += _dep_entries(manifest.get("build-dependencies", {}), "build", None)
    deps += _dep_entries(manifest.get("dev-dependencies", {}), "dev", None)
    for target_name, target_table in manifest.get("target", {}).items():
        if not isinstance(target_table, dict):
            continue
        deps += _dep_entries(target_table.get("dependencies", {}), "normal", target_name)
        deps += _dep_entries(target_table.get("build-dependencies", {}), "build", target_name)
        deps += _dep_entries(target_table.get("dev-dependencies", {}), "dev", target_name)
    return deps


def derive_metadata(crate_path: str) -> dict[str, Any]:
    """Build the NewCrate metadata object from the .crate's normalized manifest.

    The tarball produced by `cargo package` embeds a Cargo-normalized
    `Cargo.toml` -- the same manifest cargo itself would upload, with path
    dependencies already rewritten to registry dependencies.  Reading it is the
    faithful, no-guessing source for the publish metadata.
    """
    crate_bytes = read_crate_bytes(crate_path)
    with tarfile.open(fileobj=io.BytesIO(crate_bytes), mode="r:gz") as tar:
        root = _crate_root(tar)
        manifest_bytes = _extract_member(tar, root, "Cargo.toml")
        if manifest_bytes is None:
            raise PublishError(f"{crate_path}: no Cargo.toml inside the archive")
        manifest = tomllib.loads(manifest_bytes.decode("utf-8"))
        package = manifest.get("package", {})
        if not package.get("name") or not package.get("version"):
            raise PublishError(f"{crate_path}: manifest is missing a name or version")

        readme_file = package.get("readme")
        readme_content: str | None = None
        if isinstance(readme_file, str):
            readme_bytes = _extract_member(tar, root, readme_file)
            if readme_bytes is not None:
                readme_content = readme_bytes.decode("utf-8")

    features = manifest.get("features", {})
    metadata: dict[str, Any] = {
        "name": package["name"],
        "vers": package["version"],
        "deps": _collect_deps(manifest),
        "features": {name: list(values) for name, values in features.items()},
        "authors": list(package.get("authors", [])),
        "description": package.get("description"),
        "documentation": package.get("documentation"),
        "homepage": package.get("homepage"),
        "readme": readme_content,
        "readme_file": readme_file if isinstance(readme_file, str) else None,
        "keywords": list(package.get("keywords", [])),
        "categories": list(package.get("categories", [])),
        "license": package.get("license"),
        "license_file": package.get("license-file"),
        "repository": package.get("repository"),
        "badges": {},
        "links": package.get("links"),
        "rust_version": package.get("rust-version"),
    }
    return metadata


def load_metadata(crate_path: str, metadata_path: str | None) -> dict[str, Any]:
    if metadata_path is None:
        return derive_metadata(crate_path)
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except OSError as exc:
        raise PublishError(f"cannot read --metadata {metadata_path!r}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PublishError(f"--metadata {metadata_path!r} is not valid JSON: {exc}") from exc
    if not metadata.get("name") or not metadata.get("vers"):
        raise PublishError(f"--metadata {metadata_path!r} lacks a name or vers")
    return metadata


# --------------------------------------------------------------------------- #
# Framing the request body
# --------------------------------------------------------------------------- #

def encode_metadata(metadata: dict[str, Any]) -> bytes:
    return json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def build_body(metadata_json: bytes, crate_bytes: bytes) -> bytes:
    if len(metadata_json) > U32_MAX:
        raise PublishError("metadata JSON exceeds the u32 length prefix limit")
    if len(crate_bytes) > U32_MAX:
        raise PublishError(".crate tarball exceeds the u32 length prefix limit")
    return b"".join(
        (
            struct.pack("<I", len(metadata_json)),
            metadata_json,
            struct.pack("<I", len(crate_bytes)),
            crate_bytes,
        )
    )


def api_url(registry_url: str) -> str:
    return f"{registry_url.rstrip('/')}/api/v1/crates/new"


# --------------------------------------------------------------------------- #
# Sending
# --------------------------------------------------------------------------- #

def send(url: str, token: str, body: bytes, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url=url, data=body, method="PUT")
    request.add_header("Content-Type", "application/octet-stream")
    request.add_header("Accept", "application/json")
    request.add_header("Authorization", token)
    request.add_header("User-Agent", USER_AGENT)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise PublishError(
            f"registry rejected the upload: HTTP {exc.code} {exc.reason}\n{_error_detail(detail)}"
        ) from exc
    except urllib.error.URLError as exc:
        raise PublishError(f"could not reach the registry at {url}: {exc.reason}") from exc

    text = payload.decode("utf-8", "replace")
    if status is not None and not 200 <= status < 300:
        raise PublishError(f"registry returned HTTP {status}\n{_error_detail(text)}")
    try:
        parsed = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError:
        parsed = {"raw": text}
    if isinstance(parsed, dict) and parsed.get("errors"):
        raise PublishError(f"registry reported errors: {_error_detail(text)}")
    return parsed if isinstance(parsed, dict) else {"raw": text}


def _error_detail(text: str) -> str:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text.strip()
    if isinstance(parsed, dict) and isinstance(parsed.get("errors"), list):
        details = [str(item.get("detail", item)) for item in parsed["errors"]]
        return "; ".join(details)
    return text.strip()


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def command_metadata(args: argparse.Namespace) -> int:
    metadata = derive_metadata(args.crate)
    rendered = json.dumps(metadata, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
        print(f"wrote metadata for {metadata['name']} {metadata['vers']} to {args.output}")
    else:
        print(rendered)
    return 0


def command_publish(args: argparse.Namespace) -> int:
    expected = read_expected_hash(args.sha256, args.sha256_file)
    crate_bytes = read_crate_bytes(args.crate)
    actual = verify_hash(crate_bytes, expected, args.crate)

    metadata = load_metadata(args.crate, args.metadata)
    metadata_json = encode_metadata(metadata)
    body = build_body(metadata_json, crate_bytes)
    url = api_url(args.registry_url)

    if args.emit_body:
        with open(args.emit_body, "wb") as handle:
            handle.write(body)

    if args.dry_run:
        _print_dry_run(url, metadata, metadata_json, crate_bytes, actual, args)
        return 0

    token = args.token or os.environ.get("CARGO_REGISTRY_TOKEN")
    if not token:
        raise PublishError(
            "no registry token: pass --token or set CARGO_REGISTRY_TOKEN "
            "(use --dry-run to validate without a token)"
        )

    print(f"publishing {metadata['name']} {metadata['vers']} ({len(crate_bytes)} bytes) to {url}")
    result = send(url, token, body, args.timeout)
    warnings = result.get("warnings") if isinstance(result, dict) else None
    if warnings:
        for category, items in warnings.items():
            for item in items:
                print(f"warning ({category}): {item}", file=sys.stderr)
    print(f"published {metadata['name']} {metadata['vers']} (sha256 {actual})")
    return 0


def _print_dry_run(
    url: str,
    metadata: dict[str, Any],
    metadata_json: bytes,
    crate_bytes: bytes,
    actual: str,
    args: argparse.Namespace,
) -> None:
    json_len = len(metadata_json)
    crate_len = len(crate_bytes)
    token_present = bool(args.token or os.environ.get("CARGO_REGISTRY_TOKEN"))
    print("DRY RUN -- validated, nothing sent")
    print(f"  method:            PUT {url}")
    print("  header Content-Type: application/octet-stream")
    print("  header Accept:       application/json")
    print(f"  header Authorization: {'<token present>' if token_present else '<none; required for a real upload>'}")
    print(f"  header User-Agent:   {USER_AGENT}")
    print(f"  crate:             {args.crate}")
    print(f"  crate sha256:      {actual}  (matches expected)")
    print(f"  crate bytes:       {crate_len}")
    print(f"  metadata name/vers: {metadata['name']} {metadata['vers']}")
    print(f"  metadata bytes:    {json_len}")
    print("  body layout:")
    print(f"    [0:4]                        u32_le json_len   = {json_len}")
    print(f"    [4:{4 + json_len}]           metadata json")
    print(f"    [{4 + json_len}:{8 + json_len}]  u32_le crate_len  = {crate_len}")
    print(f"    [{8 + json_len}:{8 + json_len + crate_len}]  crate bytes (verbatim)")
    print(f"  total body bytes:  {8 + json_len + crate_len}")
    print(f"  metadata json:     {metadata_json.decode('utf-8')}")
    if args.emit_body:
        print(f"  body written to:   {args.emit_body}")


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="publish_crate_exact_bytes.py",
        description="Publish a pre-built .crate by its exact certified bytes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser(
        "publish", help="verify a .crate against its SHA-256 and upload it verbatim"
    )
    publish.add_argument("--crate", required=True, help="path to the pre-built .crate file")
    publish.add_argument("--sha256", help="expected SHA-256 hex digest of the .crate")
    publish.add_argument(
        "--sha256-file", help="file holding the expected SHA-256 (bare hex or a shasum line)"
    )
    publish.add_argument(
        "--metadata",
        help="path to a NewCrate metadata JSON file; if omitted it is derived "
        "from the crate's embedded normalized manifest",
    )
    publish.add_argument(
        "--registry-url",
        default=os.environ.get("MOONLAB_REGISTRY_URL", DEFAULT_REGISTRY),
        help="registry base URL (default crates.io or $MOONLAB_REGISTRY_URL); "
        "point at a local test registry to exercise without touching crates.io",
    )
    publish.add_argument(
        "--token", help="registry token (default: $CARGO_REGISTRY_TOKEN); not needed for --dry-run"
    )
    publish.add_argument(
        "--dry-run", action="store_true", help="validate and print the request without sending"
    )
    publish.add_argument(
        "--emit-body", help="write the exact request body bytes to this path (for verification)"
    )
    publish.add_argument(
        "--timeout", type=float, default=120.0, help="network timeout in seconds (default 120)"
    )
    publish.set_defaults(func=command_publish)

    metadata = subparsers.add_parser(
        "metadata", help="print the registry metadata derived from a .crate"
    )
    metadata.add_argument("--crate", required=True, help="path to the .crate file")
    metadata.add_argument("--output", help="write JSON here instead of stdout")
    metadata.set_defaults(func=command_metadata)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except PublishError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
