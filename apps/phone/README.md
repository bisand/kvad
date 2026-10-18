# Kvad for iPhone

A client of a kvad-serve, for the phone: chat, and in later phases speech,
images, video and the rest of what Kvad does. Written in Rust, with its UI
in [Slint](https://slint.dev) (`ui/*.slint`), drawn natively through
winit and Skia, and designed for iOS rather than ported from the web UI.
The plan is [`docs/phone-plan.md`](../../docs/phone-plan.md).

It is its own Cargo workspace, so Slint and Skia stay out of Kvad's builds.

The app is moving to SwiftUI over a Rust core
([`docs/phone-swiftui-plan.md`](../../docs/phone-swiftui-plan.md)). The core
is [`core/`](core): connecting, pairing, the chat, the Keychain and the
Network test, behind three C functions ([`core/kvad_core.h`](core/kvad_core.h))
and the JSON messages in [`messages.json`](messages.json). Its tests run on
the Mac and reach no server:

```bash
cargo test --manifest-path apps/phone/Cargo.toml -p kvad-phone-core
```

One more chats with a running kvad-serve, `127.0.0.1:5823` unless
`KVAD_PHONE_TEST_SERVER` says otherwise:

```bash
cargo test --manifest-path apps/phone/Cargo.toml -p kvad-phone-core -- --ignored
```

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

`CONNECT` connects at launch (an address, a paired `kvad://<node id>`, or a
pairing link, which pairs first), `TAB` opens a tab (0 Chat to 4 More),
`MODEL` picks the chat model, `SAY` sends a message, `SHEET` opens the model
sheet, and `UNPAIR` unpairs the phone from the server it connected to. `PROBE` opens the Network test, and when it is a ticket, runs it,
for as long as `PROBE_LENGTH` says (0 to 2: 30 s, 1 min, 5 min). None of it
is in a release build.

## On an iPhone

Once, on the Mac:

1. Sign in to Xcode with your Apple ID (Xcode, Settings, Accounts).
2. Put your team's id in `ios/Team.local.xcconfig`, which is not committed:

   ```
   DEVELOPMENT_TEAM = ABCDE12345
   ```

   The id is shown in Xcode's Accounts, and under Membership on
   developer.apple.com. A choice under Signing in Xcode would do too, but the
   next `xcodegen generate` throws it away. A free Personal Team works; its
   installs run out after 7 days.

Once, on the iPhone: connect it to the Mac with a cable and tap *Trust*.
Then turn on Settings, Privacy & Security, *Developer Mode*, which restarts
the phone. After the first install, a Personal Team's app also has to be
trusted under Settings, General, VPN & Device Management.

Then, with the phone unlocked:

```bash
cd apps/phone/ios && xcodegen generate && open Kvad.xcodeproj
```

Choose the iPhone at the top of the window and press Run (⌘R). Or, without
the Xcode window:

```bash
cd apps/phone/ios && xcodegen generate && xcodebuild -project Kvad.xcodeproj -scheme Kvad -destination 'generic/platform=iOS' -derivedDataPath ../target/xcode-device -allowProvisioningUpdates build
```

```bash
xcrun devicectl device install app --device <iPhone UDID> apps/phone/target/xcode-device/Build/Products/Debug-iphoneos/Kvad.app
```

`xcrun devicectl list devices` shows the UDID. "Signing for Kvad requires a
development team" means `Team.local.xcconfig` is missing; "needs to be
unlocked" means what it says. The Keychain entitlement that `project.yml`
asks for is in the signature `build.sh` makes, so pairings are kept on the
device as they are in the simulator.

## From anywhere, through the tunnel

On the Mac, with `[remote]` on in kvad-serve's `kvad.toml`:

```bash
kvad remote pair
```

Point the iPhone's Camera at the QR code it shows and tap the `kvad://`
link it offers: Kvad opens with the link in the Server field and says which
server it is for. Or copy the link (Universal Clipboard carries it to the
phone) and tap *Paste a pairing link*. Either way nothing happens until you
tap *Connect*: a link only fills in the field. The phone pairs, through the relay
if it is not on the Mac's network, and keeps its device key, the key the
server made for it, and its relay voucher in the Keychain: this phone only,
while unlocked, never synced. After that the Server field holds
`kvad://<node id>`, and *Connect* reaches the Mac from Wi-Fi or mobile data
alike. More shows which path it is on, and *Unpair this phone* revokes the
pairing on the Mac as well as here.

To try an opened link in the simulator, which has no camera:

```bash
xcrun simctl openurl booted 'kvad://…&pair=…'
```

Each time the app comes back to the foreground it tells the tunnel that the
network may have changed, so that a phone that left the Wi-Fi while Kvad was
in the background looks for its paths again at once.

Plain HTTP still reaches only the phone's own network (App Transport
Security's `NSAllowsLocalNetworking`), so an address typed in works on the
LAN alone.

## Network test

More, and the Connect screen, open a Network test: `relay-probe dial` from
[`tools/relay-probe`](../../tools/relay-probe/README.md), run by the phone
itself. It is the probe's own library, so the phone dials as the laptop
does: X25519MLKEM768 only, the listener's key pinned by the ticket, no port
mapping, and the same path names and table. It is what gives Phase 0 of
[`docs/relay-plan.md`](../../docs/relay-plan.md) its phone column: on
mobile data the phone is behind the carrier's NAT alone, where a laptop on
its hotspot is behind two.

The phone's probe key is made on first use in the app's container
(`Library/Application Support/kvad-phone/probe.key`), until the app has the
Keychain. The screen shows its id, and Copy id puts it on the clipboard,
which Universal Clipboard hands to the Mac. A run keeps the screen from
locking, since a locked phone suspends the app and its sockets.

In the simulator, against a listener on this Mac: [running](../../docs/phone/network-test-running.png),
[done](../../docs/phone/network-test-done.png), [its log](../../docs/phone/network-test-log.png),
and [the way in from More](../../docs/phone/network-test-more.png).

### The run over mobile data

1. On the iPhone: Kvad, Network test (on the Connect screen, or under More),
   Copy id.
2. On the Mac, add the id to the relay's allow list, with a comment, and
   deploy the relay again (see `tools/kvad-relay/deploy/README.md`):

   ```bash
   pbpaste
   ```

   ```bash
   cd tools/kvad-relay && RELAY_MANIFEST=deploy/k8s-traefik.yaml RELAY_HOSTNAME=relay2.kvad.eu RELAY_ADDRESS=217.170.199.230 deploy/deploy.sh root@tst01 deploy/allow.local -i ~/.ssh/id_tst01_networco_bogentech_no
   ```

3. On the Mac, at home on its usual network:

   ```bash
   tools/relay-probe/target/release/relay-probe listen --allow <the phone's id> --relay https://relay2.kvad.eu
   ```

   Copy the ticket it prints (select it, ⌘C).
4. On the iPhone: Wi-Fi off in Control Centre, so that mobile data is the
   only network. Paste from the Mac, choose a length, Start, and leave the
   app in front until it says Done.
5. Note the `network:` lines from both ends, the path changes in the log,
   and the table, as for the laptop's runs. A screenshot of the screen
   holds all of the phone's side.

Worth a second run each: 5 min, for a direct path that comes late or a
mapping that expires; and Wi-Fi on at home, for the LAN case.

## Licences

Kvad's code here is MIT. Slint is used under its royalty-free licence, whose
attribution is the About section of the More tab (`AboutSlint`).
