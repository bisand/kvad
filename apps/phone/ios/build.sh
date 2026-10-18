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

cargo_profile=$([[ $profile == debug ]] && echo dev || echo release)

# The entitlements (the Keychain's access group, see project.yml) reach the
# app differently on each. The simulator reads them from two sections of the
# executable, which Xcode's own link would have added, so cargo's link adds
# them: `cargo rustc` passes the flags to the final link alone, so nothing it
# depends on is built again. A device reads them from the signature, below.
xcent="${TARGET_TEMP_DIR:-}/${FULL_PRODUCT_NAME:-Kvad.app}-Simulated.xcent"
if [[ "$target" == aarch64-apple-ios-sim && -f "$xcent" ]]; then
    cargo rustc --profile "$cargo_profile" --target "$target" --bin "$1" -- \
        -C "link-arg=-Wl,-sectcreate,__TEXT,__entitlements,$xcent" \
        -C "link-arg=-Wl,-sectcreate,__TEXT,__ents_der,$xcent.der"
else
    cargo build --profile "$cargo_profile" --target "$target" --bin "$1"
fi
cp "target/$target/$profile/$1" "$TARGET_BUILD_DIR/$EXECUTABLE_PATH"

# A device build is signed again, now that the binary has been replaced, with
# the entitlements Xcode prepared for it.
if [[ "$target" == aarch64-apple-ios && -n "${EXPANDED_CODE_SIGN_IDENTITY:-}" && "${CODE_SIGNING_ALLOWED:-YES}" != NO ]]; then
    entitlements="${TARGET_TEMP_DIR:-}/${FULL_PRODUCT_NAME:-Kvad.app}.xcent"
    if [[ -s "$entitlements" ]]; then
        codesign --force --sign "$EXPANDED_CODE_SIGN_IDENTITY" --entitlements "$entitlements" "$TARGET_BUILD_DIR/$EXECUTABLE_PATH"
    else
        codesign --force --sign "$EXPANDED_CODE_SIGN_IDENTITY" "$TARGET_BUILD_DIR/$EXECUTABLE_PATH"
    fi
fi
