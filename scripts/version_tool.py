#!/usr/bin/env python3
"""Synchronize and verify Moonlab's release version surfaces."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SEMVER_RE = re.compile(
    r"^(?P<base>\d+\.\d+\.\d+)(?:-(?P<kind>alpha|beta|rc)\.(?P<num>\d+))?$"
)
JS_PACKAGES = ("core", "algorithms", "viz", "react", "vue")
RUST_MANIFESTS = ("moonlab", "moonlab-sys", "moonlab-tui")
RUST_PATH_DEPENDENCIES = (
    ("moonlab/Cargo.toml", "moonlab-sys"),
    ("moonlab-tui/Cargo.toml", "moonlab"),
)
JS_VERSION_TESTS = (
    "react/src/__tests__/exports.test.ts",
    "viz/src/__tests__/utilities.test.ts",
    "vue/src/__tests__/exports.test.ts",
)
VERSION_DEFAULTS = (
    (
        "Debian package default",
        "scripts/build-deb.sh",
        r'^VERSION="\$\{1:-([^}]+)\}"$',
        1,
    ),
    (
        "Ubuntu release container",
        "docker/ubuntu/release/Dockerfile",
        r"^ARG VERSION=([^\s]+)$",
        1,
    ),
    (
        "Debian release container",
        "docker/debian/release/Dockerfile",
        r"^ARG VERSION=([^\s]+)$",
        1,
    ),
    (
        "Debian debug container",
        "docker/debian/debug/Dockerfile",
        r"^ARG VERSION=([^\s]+)$",
        1,
    ),
    (
        "Compose container defaults",
        "docker/docker-compose.yml",
        r"VERSION: \$\{VERSION:-([^}]+)\}",
        3,
    ),
)


def pep440_version(version: str) -> str:
    match = SEMVER_RE.fullmatch(version)
    if not match:
        raise ValueError(
            "version must be MAJOR.MINOR.PATCH or a numbered "
            "-alpha.N, -beta.N, or -rc.N prerelease"
        )
    if not match.group("kind"):
        return match.group("base")
    marker = {"alpha": "a", "beta": "b", "rc": "rc"}[match.group("kind")]
    return f"{match.group('base')}{marker}{match.group('num')}"


def replace_one(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text()
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"could not update the expected version field in {path}")
    path.write_text(updated)


def replace_version_matches(
    path: Path, pattern: str, version: str, expected_count: int
) -> None:
    text = path.read_text()
    updated, count = re.subn(
        pattern,
        lambda match: match.group(0).replace(match.group(1), version, 1),
        text,
        flags=re.MULTILINE,
    )
    if count != expected_count:
        raise RuntimeError(
            f"expected {expected_count} version fields in {path}, found {count}"
        )
    path.write_text(updated)


def set_version(version: str) -> None:
    python_version = pep440_version(version)
    (ROOT / "VERSION.txt").write_text(f"{version}\n")

    replace_one(
        ROOT / "bindings/python/pyproject.toml",
        r'^version = "[^"]+"$',
        f'version = "{python_version}"',
    )
    replace_one(
        ROOT / "bindings/python/moonlab/__init__.py",
        r'^__version__ = "[^"]+"$',
        f'__version__ = "{python_version}"',
    )

    for crate in RUST_MANIFESTS:
        replace_one(
            ROOT / f"bindings/rust/{crate}/Cargo.toml",
            r'^version = "[^"]+"$',
            f'version = "{version}"',
        )
    for manifest, dependency in RUST_PATH_DEPENDENCIES:
        replace_one(
            ROOT / f"bindings/rust/{manifest}",
            rf'^({re.escape(dependency)} = \{{ version = ")[^"]+(".*)$',
            rf'\g<1>{version}\g<2>',
        )

    package_files = [ROOT / "bindings/javascript/package.json"]
    package_files.extend(
        ROOT / f"bindings/javascript/packages/{package}/package.json"
        for package in JS_PACKAGES
    )
    package_files.append(ROOT / "bindings/javascript/demo/package.json")
    for path in package_files:
        payload = json.loads(path.read_text())
        payload["version"] = version
        path.write_text(json.dumps(payload, indent=2) + "\n")

    for package in JS_PACKAGES:
        replace_one(
            ROOT / f"bindings/javascript/packages/{package}/src/index.ts",
            r"^export const VERSION = '[^']+';$",
            f"export const VERSION = '{version}';",
        )
    for relative_path in JS_VERSION_TESTS:
        replace_one(
            ROOT / f"bindings/javascript/packages/{relative_path}",
            r"expect\(VERSION\)\.toBe\('[^']+'\);",
            f"expect(VERSION).toBe('{version}');",
        )
    replace_one(
        ROOT / "bindings/javascript/packages/core/emscripten/post.js",
        r"^  core: '[^']+',$",
        f"  core: '{version}',",
    )
    replace_one(
        ROOT / "README.md",
        r"version-[^-\s]+(?:-[^-\s]+)*-blue",
        f"version-{version}-blue",
    )
    for _label, relative_path, pattern, expected_count in VERSION_DEFAULTS:
        replace_version_matches(
            ROOT / relative_path, pattern, version, expected_count
        )

    print(f"Moonlab versions synchronized: semver={version} pypi={python_version}")


def first_match(path: Path, pattern: str) -> str:
    match = re.search(pattern, path.read_text(), flags=re.MULTILINE)
    if not match:
        return "<missing>"
    return match.group(1)


def check_version(tag: str | None) -> None:
    version = (ROOT / "VERSION.txt").read_text().strip()
    python_version = pep440_version(version)
    errors: list[str] = []

    def expect(label: str, actual: str, expected: str) -> None:
        if actual != expected:
            errors.append(f"{label}: expected {expected!r}, found {actual!r}")

    if tag:
        expect("release tag", tag, f"v{version}")

    expect(
        "Python project",
        first_match(ROOT / "bindings/python/pyproject.toml", r'^version = "([^"]+)"$'),
        python_version,
    )
    expect(
        "Python runtime",
        first_match(
            ROOT / "bindings/python/moonlab/__init__.py",
            r'^__version__ = "([^"]+)"$',
        ),
        python_version,
    )

    for crate in RUST_MANIFESTS:
        expect(
            f"Rust {crate}",
            first_match(
                ROOT / f"bindings/rust/{crate}/Cargo.toml",
                r'^version = "([^"]+)"$',
            ),
            version,
        )
    for manifest, dependency in RUST_PATH_DEPENDENCIES:
        expect(
            f"Rust dependency {dependency} in {manifest}",
            first_match(
                ROOT / f"bindings/rust/{manifest}",
                rf'^{re.escape(dependency)} = \{{ version = "([^"]+)".*$',
            ),
            version,
        )

    package_files = [ROOT / "bindings/javascript/package.json"]
    package_files.extend(
        ROOT / f"bindings/javascript/packages/{package}/package.json"
        for package in JS_PACKAGES
    )
    package_files.append(ROOT / "bindings/javascript/demo/package.json")
    for path in package_files:
        expect(
            f"npm {path.parent.name}",
            str(json.loads(path.read_text()).get("version", "<missing>")),
            version,
        )

    for package in JS_PACKAGES:
        expect(
            f"JavaScript runtime {package}",
            first_match(
                ROOT / f"bindings/javascript/packages/{package}/src/index.ts",
                r"^export const VERSION = '([^']+)';$",
            ),
            version,
        )
    for relative_path in JS_VERSION_TESTS:
        expect(
            f"JavaScript version test {relative_path}",
            first_match(
                ROOT / f"bindings/javascript/packages/{relative_path}",
                r"expect\(VERSION\)\.toBe\('([^']+)'\);",
            ),
            version,
        )
    expect(
        "WASM runtime",
        first_match(
            ROOT / "bindings/javascript/packages/core/emscripten/post.js",
            r"^  core: '([^']+)',$",
        ),
        version,
    )
    for label, relative_path, pattern, expected_count in VERSION_DEFAULTS:
        matches = re.findall(
            pattern, (ROOT / relative_path).read_text(), re.MULTILINE
        )
        if len(matches) != expected_count:
            errors.append(
                f"{label}: expected {expected_count} version fields, found {len(matches)}"
            )
            continue
        for index, actual in enumerate(matches, start=1):
            expect(f"{label} #{index}", actual, version)

    if errors:
        print("Release version gate failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Release version gate passed: semver={version} pypi={python_version}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    set_parser = subparsers.add_parser("set")
    set_parser.add_argument("version")
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--tag")
    args = parser.parse_args()

    if args.command == "set":
        set_version(args.version)
    else:
        check_version(args.tag)


if __name__ == "__main__":
    main()
