#!/usr/bin/env bash
# Xcode's build phase for the app: builds the Rust binary for the platform
# Xcode is building for, and puts it where Xcode expects the executable.
# Takes the cargo binary's name.
set -euo pipefail

export PATH="$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
cd "$(dirname "$0")/.."

# macOS's bash is 3.2, where an empty array under `set -u` is unbound.
profile=debug
if [[ "${CONFIGURATION:-Debug}" != "Debug" ]]; then
    profile=release
fi

# The simulator and the device are different targets for the same arm64.
case "${PLATFORM_NAME:-iphonesimulator}" in
    iphonesimulator) target=aarch64-apple-ios-sim ;;
    iphoneos) target=aarch64-apple-ios ;;
    *) echo "build.sh: no Rust target for ${PLATFORM_NAME}" >&2; exit 1 ;;
esac

cargo build --profile "$([[ $profile == debug ]] && echo dev || echo release)" --target "$target" --bin "$1"
cp "target/$target/$profile/$1" "$TARGET_BUILD_DIR/$EXECUTABLE_PATH"

# A device build is signed again, now that the binary has been replaced.
if [[ "$target" == aarch64-apple-ios && -n "${EXPANDED_CODE_SIGN_IDENTITY:-}" && "${CODE_SIGNING_ALLOWED:-YES}" != NO ]]; then
    codesign --force --sign "$EXPANDED_CODE_SIGN_IDENTITY" "$TARGET_BUILD_DIR/$EXECUTABLE_PATH"
fi
