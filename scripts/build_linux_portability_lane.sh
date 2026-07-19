#!/usr/bin/env bash
# Produce one fail-closed Moonlab Linux portability evidence lane in a clean container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="${MOONLAB_PORTABILITY_OUTPUT_DIR:-/evidence}"
BUILD_DIR="${MOONLAB_PORTABILITY_BUILD_DIR:-/tmp/moonlab-portability-build}"
INSTALL_DIR="${MOONLAB_PORTABILITY_INSTALL_DIR:-/tmp/moonlab-portability-install}"
JOBS="${MOONLAB_PORTABILITY_JOBS:-2}"

required_environment=(
    MOONLAB_PORTABILITY_PROFILE_ID
    MOONLAB_PORTABILITY_ARCHITECTURE
    MOONLAB_PORTABILITY_IMAGE_DIGEST
    MOONLAB_PORTABILITY_SOURCE_HEAD
    MOONLAB_PORTABILITY_SOURCE_TREE
    MOONLAB_PORTABILITY_SOURCE_FINGERPRINT
)
for name in "${required_environment[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "build-linux-portability-lane: required environment variable $name is missing" >&2
        exit 2
    fi
done

case "$(uname -m)" in
    x86_64) detected_architecture=amd64 ;;
    aarch64|arm64) detected_architecture=arm64 ;;
    *)
        echo "build-linux-portability-lane: unsupported architecture $(uname -m)" >&2
        exit 2
        ;;
esac
if [[ "$detected_architecture" != "$MOONLAB_PORTABILITY_ARCHITECTURE" ]]; then
    echo "build-linux-portability-lane: profile requires $MOONLAB_PORTABILITY_ARCHITECTURE, container is $detected_architecture" >&2
    exit 2
fi

case "$JOBS" in
    ''|*[!0-9]*|0)
        echo "build-linux-portability-lane: job count must be a positive integer" >&2
        exit 2
        ;;
esac

for temporary_path in "$BUILD_DIR" "$INSTALL_DIR"; do
    case "$temporary_path" in
        /tmp/?*) ;;
        *)
            echo "build-linux-portability-lane: temporary paths must be non-root children of /tmp: $temporary_path" >&2
            exit 2
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"
if find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "build-linux-portability-lane: output directory must be empty: $OUTPUT_DIR" >&2
    exit 2
fi

rm -rf -- "$BUILD_DIR" "$INSTALL_DIR"

"$SCRIPT_DIR/build-linux.sh" \
    --build-dir "$BUILD_DIR" \
    --jobs "$JOBS" \
    2>&1 | tee "$OUTPUT_DIR/build.log"

ctest --test-dir "$BUILD_DIR" \
    --output-on-failure \
    --output-junit "$OUTPUT_DIR/test-results.xml" \
    --timeout 180 \
    -L "(core|abi|topology|health)" \
    -LE "long|memory_heavy" \
    --parallel "$JOBS" \
    2>&1 | tee "$OUTPUT_DIR/test.log"

cmake --install "$BUILD_DIR" --prefix "$INSTALL_DIR"
tar -czf "$OUTPUT_DIR/install-tree.tar.gz" -C "$INSTALL_DIR" .

"$SCRIPT_DIR/package_release_artifact.sh" \
    --build-dir "$BUILD_DIR" \
    --output "$OUTPUT_DIR/package.tar.gz"

"$SCRIPT_DIR/verify_release_package.sh" \
    --package "$OUTPUT_DIR/package.tar.gz" \
    --cmake-log "$OUTPUT_DIR/cmake-consumer.log" \
    --pkg-config-log "$OUTPUT_DIR/pkg-config-consumer.log" \
    --require-pkg-config

python3 "$SCRIPT_DIR/moonlab_linux_portability.py" \
    --profiles "$REPO_ROOT/release/linux_portability_profiles.v1.json" \
    --emit-lane-profile "$MOONLAB_PORTABILITY_PROFILE_ID" \
    --image-digest "$MOONLAB_PORTABILITY_IMAGE_DIGEST" \
    --source-head "$MOONLAB_PORTABILITY_SOURCE_HEAD" \
    --source-tree "$MOONLAB_PORTABILITY_SOURCE_TREE" \
    --source-fingerprint "$MOONLAB_PORTABILITY_SOURCE_FINGERPRINT" \
    --test-results "$OUTPUT_DIR/test-results.xml" \
    --artifact "build_log=$OUTPUT_DIR/build.log" \
    --artifact "test_log=$OUTPUT_DIR/test.log" \
    --artifact "install_tree=$OUTPUT_DIR/install-tree.tar.gz" \
    --artifact "package=$OUTPUT_DIR/package.tar.gz" \
    --artifact "cmake_consumer_log=$OUTPUT_DIR/cmake-consumer.log" \
    --artifact "pkg_config_consumer_log=$OUTPUT_DIR/pkg-config-consumer.log" \
    --out "$OUTPUT_DIR/lane-$MOONLAB_PORTABILITY_PROFILE_ID.json"

echo "build-linux-portability-lane: PASS ($MOONLAB_PORTABILITY_PROFILE_ID)"
