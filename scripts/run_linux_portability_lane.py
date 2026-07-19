#!/usr/bin/env python3
"""Run one canonical Moonlab Linux portability lane in a clean Docker image."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from moonlab_linux_portability import _load_profiles, _read_json, _validate_lane
from moonlab_source_identity import source_identity


ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "release/linux_portability_profiles.v1.json"
IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class LaneRunError(RuntimeError):
    """Raised when a canonical portability lane cannot be proven."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


def _run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError as exc:
        raise LaneRunError(f"required command is unavailable: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise LaneRunError(f"command failed ({' '.join(command)}): {detail}") from exc


def _source_identity() -> tuple[str, str, str]:
    identity = source_identity(ROOT)
    if identity["dirty"]:
        raise LaneRunError("portability evidence requires a clean exact worktree")
    return (
        str(identity["git_head"]),
        str(identity["git_tree"]),
        str(identity["source_fingerprint"]),
    )


def _resolved_image_digest(image: str) -> str:
    _run(["docker", "pull", image])
    result = _run(
        ["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", image],
        capture=True,
    )
    try:
        repo_digests = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise LaneRunError(f"docker returned invalid RepoDigests JSON for {image}") from exc
    digests = sorted({value.rsplit("@", 1)[-1] for value in repo_digests if isinstance(value, str) and "@" in value})
    if len(digests) != 1 or IMAGE_DIGEST.fullmatch(digests[0]) is None:
        raise LaneRunError(f"image {image} did not resolve to exactly one immutable sha256 digest")
    return digests[0]


def run_lane(profile_id: str, output: Path, jobs: int) -> Path:
    profiles, _ = _load_profiles(PROFILES)
    profile = next((item for item in profiles if item["id"] == profile_id), None)
    if profile is None:
        raise LaneRunError(f"unknown canonical portability profile {profile_id}")
    if jobs <= 0:
        raise LaneRunError("--jobs must be a positive integer")

    output = output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise LaneRunError("--output must be outside the source checkout so the bound source stays clean")
    if output.exists():
        raise LaneRunError(f"output path already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    head, tree, fingerprint = _source_identity()
    image_digest = _resolved_image_digest(profile["image"])

    with tempfile.TemporaryDirectory(prefix=f".{profile_id}.", dir=output.parent) as temporary_name:
        temporary = Path(temporary_name)
        command = [
            "docker",
            "run",
            "--rm",
            "--platform",
            f"linux/{profile['architecture']}",
            "--volume",
            f"{ROOT}:/src:ro",
            "--volume",
            f"{temporary}:/evidence",
            "--workdir",
            "/src",
            "--env",
            f"MOONLAB_PORTABILITY_PROFILE_ID={profile_id}",
            "--env",
            f"MOONLAB_PORTABILITY_ARCHITECTURE={profile['architecture']}",
            "--env",
            f"MOONLAB_PORTABILITY_IMAGE_DIGEST={image_digest}",
            "--env",
            f"MOONLAB_PORTABILITY_SOURCE_HEAD={head}",
            "--env",
            f"MOONLAB_PORTABILITY_SOURCE_TREE={tree}",
            "--env",
            f"MOONLAB_PORTABILITY_SOURCE_FINGERPRINT={fingerprint}",
            "--env",
            f"MOONLAB_PORTABILITY_JOBS={jobs}",
            profile["image"],
            "/bin/sh",
            "-c",
            "exec bash scripts/build_linux_portability_lane.sh",
        ]
        _run(command)

        lane_path = temporary / f"lane-{profile_id}.json"
        lane = _validate_lane(_read_json(lane_path, "produced lane manifest"), profile)
        expected_source = {
            "git_head": head,
            "git_tree": tree,
            "source_fingerprint": fingerprint,
            "dirty": False,
        }
        if lane["source"] != expected_source or lane["image"]["digest"] != image_digest:
            raise LaneRunError("produced lane manifest does not bind the resolved source and image")
        os.replace(temporary, output)

    return output / f"lane-{profile_id}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    arguments = parser.parse_args()
    try:
        lane = run_lane(arguments.profile, arguments.output, arguments.jobs)
    except (LaneRunError, OSError, ValueError) as exc:
        print(f"run-linux-portability-lane: {exc}", file=os.sys.stderr)
        return 2
    print(f"run-linux-portability-lane: PASS ({lane})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
