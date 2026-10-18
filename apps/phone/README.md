# Kvad for iPhone

A client of a kvad-serve, for the phone: chat, and in later phases speech,
images, video and the rest of what Kvad does. Written in Rust, with its UI
in [Slint](https://slint.dev) (`ui/*.slint`), drawn natively through
winit and Skia, and designed for iOS rather than ported from the web UI.
The plan is [`docs/phone-plan.md`](../../docs/phone-plan.md).

It is its own Cargo workspace, so Slint and Skia stay out of Kvad's builds.

## On the Mac, while designing

```bash
cargo run --manifest-path apps/phone/Cargo.toml
```

It opens a window at an iPhone's size in points. Type is smaller than on the
phone, because the Mac has no Dynamic Type to set 1rem. The simulator is the
check that counts.

## In the iOS Simulator

Once: `rustup target add aarch64-apple-ios-sim aarch64-apple-ios`, and
XcodeGen (`brew install xcodegen`).

```bash
cd apps/phone/ios && xcodegen generate && open Kvad.xcodeproj
```

Or from the command line:

```bash
cd apps/phone/ios && xcodegen generate && xcodebuild -project Kvad.xcodeproj -scheme Kvad -sdk iphonesimulator -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -derivedDataPath ../target/xcode build
```

```bash
xcrun simctl install booted apps/phone/target/xcode/Build/Products/Debug-iphonesimulator/Kvad.app && xcrun simctl launch booted eu.kvad.phone
```

The simulator shares the Mac's network, so a kvad-serve on this Mac is at
`127.0.0.1:5823`, the default the app offers.

A debug build reads a script from its environment, for screenshots and for
looking at a state without tapping to it. `simctl launch` passes
`SIMCTL_CHILD_KVAD_PHONE_*` through:

```bash
SIMCTL_CHILD_KVAD_PHONE_CONNECT=127.0.0.1:5823 SIMCTL_CHILD_KVAD_PHONE_TAB=3 xcrun simctl launch booted eu.kvad.phone
```

`CONNECT` connects at launch, `TAB` opens a tab (0 Chat to 4 More), `MODEL`
picks the chat model, `SAY` sends a message, and `SHEET` opens the model
sheet. None of it is in a release build.

## On an iPhone

Xcode, with your own team chosen under Signing for the Kvad target. Plain
HTTP reaches only the phone's own network (App Transport Security's
`NSAllowsLocalNetworking`), so the Mac's LAN address works and the internet
does not, until the tunnel in `docs/relay-plan.md` is in.

## Licences

Kvad's code here is MIT. Slint is used under its royalty-free licence, whose
attribution is the About section of the More tab (`AboutSlint`).
