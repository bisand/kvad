#!/usr/bin/env bash
# Xcode's build phase for the app, before it compiles the Swift: builds the
# Rust core (apps/phone/core) as a static library for the platform Xcode is
# building for, and links it where the target's LIBRARY_SEARCH_PATHS look.
set -euo pipefail

export PATH="$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
cd "$(dirname "$0")/.."

profile=debug
if [[ "${CONFIGURATION:-Debug}" != "Debug" ]]; then
    profile=release
fi
cargo_profile=$([[ $profile == debug ]] && echo dev || echo release)

# The simulator and the device are different targets for the same arm64.
case "${PLATFORM_NAME:-iphonesimulator}" in
    iphonesimulator) target=aarch64-apple-ios-sim ;;
    iphoneos) target=aarch64-apple-ios ;;
    *) echo "build.sh: no Rust target for ${PLATFORM_NAME}" >&2; exit 1 ;;
esac

cargo build --profile "$cargo_profile" --target "$target" -p kvad-phone-core

# A link, not a copy: a debug library is some 430 MB, nearly all debug info.
out="${TARGET_TEMP_DIR:?build.sh runs from Xcode}/rust"
mkdir -p "$out"
ln -sf "$PWD/target/$target/$profile/libkvad_phone_core.a" "$out/libkvad_phone_core.a"
