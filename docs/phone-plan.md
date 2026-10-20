# Plan: Kvad on the iPhone

Written 2026-10-17. Tracked with the remote access work in
[#240](https://github.com/bisand/kvad/issues/240), which it exists to test,
and which it outgrows: the app is a client for everything Kvad does, not a
test harness for a tunnel.

## What the owner asked for

An iPhone app that is a client of a Kvad server and can do what Kvad does
from the phone: chat, transcribe, make images and videos, manage models,
watch jobs. It should be written in Rust, using a framework that brings it
to iOS. And it should *look like a phone app*. The web UI stays what the
server serves to a browser. The phone gets its own UI, designed for the
phone, and well made.

## The choices

**SwiftUI over a Rust core.** The screens are SwiftUI, so scrolling, the
keyboard, the tab bar and text are UIKit's own. Everything under them is
Rust: the client, pairing, the tunnel, the Keychain and the Network test,
in `apps/phone/core`. Swift calls it through four C functions, with one
JSON message each way and the microphone's samples as they are. [phone-swiftui-plan.md](phone-swiftui-plan.md) has
the boundary and why it is drawn there.

The app began on Slint 1.18, Rust only, which the owner chose over Dioxus
(whose iOS UI is a web view) and Tauri (whose UI would have been the Svelte
web UI). A day on a real phone showed how much UIKit gives every app that
Slint, which draws every pixel itself, had to rebuild by hand: momentum and
bounce, the keyboard's frame, wrapped text's height. That list is in
phone-swiftui-plan.md, under *Why the move*.

**Its own workspace.** `apps/phone/` is a Cargo workspace of its own, as
`tools/relay-probe` is, so that the core's iroh and the iOS targets never
enter Kvad's builds or its CI time. The core has no UI and no iOS-only
code beyond the Keychain, so an Android app can call the same messages
over JNI later, with a UI in Kotlin.

**Not the CLI's client, yet.** `crates/llm/src/client.rs` is the CLI's
client of a server, but it lives in the engine crate, among tokenizers and
safetensors that a phone has no use for. The app starts with a small client
of its own (`apps/phone/src/api.rs`), on the same `ureq` and the same
server-sent events. Once both are stable, the shared part moves into a
`kvad-client` crate that both depend on, so that the CLI and the phone
cannot drift apart on the protocol.

## How it is built

```
apps/phone/
  Cargo.toml          its own workspace, of the core alone
  core/               the Rust: a static library for Xcode
    kvad_core.h       the four C functions
    src/lib.rs        the C functions: commands in, events out
    src/session.rs    connect, pair, chat, unpair, opened links
    src/api.rs        the HTTP client: health, models, chat as a stream
    src/pairings.rs   the paired servers and the device key
    src/secrets.rs    the one Keychain item they are kept in
    src/probe.rs      the Network test, on relay-probe's dialler
    src/speak.rs      Speak: the take's file, live over /v1/realtime, the resume
    tests/messages.rs the core driven by its messages alone
    tests/speak.rs    Speak against a scripted realtime server
  messages.json       every message, which both sides' tests read
  ios/project.yml     XcodeGen: the app and its test target
  ios/build.sh        Xcode's first build phase: cargo for the right target
  ios/Kvad/           the SwiftUI app
  ios/KvadTests/      every event in messages.json decodes
```

- **Threads.** SwiftUI runs on the main thread. The core runs each command
  that waits on the network on a thread of its own, holds the tunnel's
  tokio runtime, and calls back on whichever thread has the answer; Swift
  moves each event onto the main actor before the model changes.
- **The transport is pluggable.** `ureq` 3 takes its connection from a
  `Transport`. Plain TCP and TLS came first. The tunnel from
  `docs/relay-plan.md` is a second `Transport`, so that the screens never
  know how the bytes travel.

## The design

Apple's Human Interface Guidelines are the brief. The web UI's layout is
not.

- **A tab bar at the bottom**, five tabs as iOS allows: **Chat**, **Speak**
  (transcription), **Create** (images and videos), **Models**, and **More**
  for everything else.
- **Large titles** that shrink when the content scrolls, **grouped lists**
  with inset rounded sections for settings and model lists, and **chat
  bubbles** with the reply streaming in word by word.
- **The system's type and colours.** San Francisco through the system font,
  sizes that follow Dynamic Type, light and dark from the system setting,
  and semantic colours (label, secondary label, grouped background,
  separator) rather than fixed greys. One accent colour, Kvad's.
- **The phone's geometry.** Content inside the safe area, the composer
  riding on top of the keyboard (SwiftUI's own keyboard avoidance), and touch
  targets of at least 44 points.
- **Motion that means something:** a pressed state on every control, short
  transitions between screens, and a streaming caret while a reply is being
  written. No decoration for its own sake.
- **States designed, not left over:** first launch (no server yet), the
  server not answering, signed out, nothing loaded, a model loading with its
  progress, and an error that says what to do next.

## What it does, screen by screen

| Screen | Server API | Phase |
|---|---|---|
| Connect: a server URL, an API key, then the health check | `/api/health`, `/api/auth` | 1 |
| Chat: pick a model, converse, the reply streamed | `/v1/models`, `/v1/chat/completions` (SSE) | 1 |
| Models: what is on disk and loaded; load, unload, pull, search | `/api/models*` | 1 (list), 3 (actions) |
| Speak: record, then transcribe a file; then live words back while speaking | `/v1/audio/transcriptions`, the live WebSocket in `docs/speech-plan.md` | 2 |
| Create: images (generate, edit with a mask) and videos, with job progress and a gallery | `/v1/images/*`, `/v1/videos*`, `/api/images*`, `/api/jobs*` | 3 |
| Pairing: scan the Mac's QR code, the PAKE code, then the tunnel | `docs/relay-plan.md` | 4 |
| More: jobs, queue, training and tuning, evals, benchmarks, datasets, monitoring, cluster, sessions and keys, settings | the rest of `/api/*` | 5 |

Some of the server's administration (users, restarting it, moving its data
directory) is reachable only when the device's scope allows it, as
`docs/relay-plan.md` says under *Least privilege for devices*. The More tab
shows what the key may do, and nothing else.

## Security

The app follows the relay plan's *Security* section, and on the phone that
means:

- **The API key and the device key live in the Keychain**, accessible only
  when the device is unlocked, kept to this device only, never synced and
  never backed up. Never in a file or in the app's defaults.
- **No plain HTTP off the phone's own network.** App Transport Security
  stays on. `NSAllowsLocalNetworking` lets the app reach a Mac on the LAN
  during development, and the tunnel replaces that. HTTPS is verified
  against the system's roots, and a tunnel connection against the pinned key.
- **Nothing leaves except to the owner's server.** No analytics, no crash
  reporter, no third-party SDK. The app makes no network request the owner
  did not point it at.
- **Recordings and transcripts** are kept with Data Protection class
  Complete, and are deleted from the phone once the server has them, unless
  the owner keeps them.
- **A privacy screen.** The app switcher's snapshot shows the app's mark,
  not the last transcript or chat. Face ID to open the app is an option.
- **Permissions asked when needed, with a reason:** the microphone on the
  first recording, the camera on the first QR scan, and the local network
  on the first LAN connection.

## Phases

**Phase 1: the shell, in the simulator.** The workspace, the Xcode project
through XcodeGen, the tab bar and the theme. Connect to a kvad-serve by URL
and key, list the models, and chat with a streamed reply. Test: the app in
the iOS Simulator against the kvad-serve on this Mac, which the simulator
reaches at `127.0.0.1:5823`. A screenshot of each state goes in the PR.
`api.rs` gets host-side tests against a fake server, as `client.rs` has.

*Built, 2026-10-17:* `apps/phone`, running in the iOS 26.5 simulator on an
iPhone 17 Pro against the kvad-serve on this Mac. It covers Connect (with
the server's address remembered and the key not), the Models list grouped
by kind, Chat with the reply streamed, a model sheet, and More with the
server's details and the Slint attribution. Light and dark both follow the
system. Speak and Create are designed empty states. Not done yet: the key
in the Keychain (until then it is held in memory only), pressing Stop
against a long reply, the keyboard's frame (it needs a person typing, which
a debug launch script cannot do), and a real device.

**Phase 2: Speak.** Its own plan, [phone-speak-plan.md](phone-speak-plan.md).
The microphone is Swift's `AVAudioEngine` rather than `cpal`, since the
work on iOS is the audio session's and nothing else's. The live session and
the take kept on the phone are the core's. Step 1, the core, is built.

**Phase 3: Create, and model management.** Images and videos with job
progress over SSE. Loading and pulling models with their progress.

**Phase 4: the tunnel.** Pairing by QR code: the camera through a small
`objc2` binding to AVFoundation's code scanner (in the end the Camera app's,
then, with SwiftUI, VisionKit's in the app: below). Then iroh as a `Transport`,
the keys in the Keychain, and the path shown in the connection's status
(`lan`, `direct`, `relay`). When iOS reports a network change (`NWPathMonitor`), the app tells the
endpoint at once (`network_change`), so that a phone moving from Wi-Fi to 5G
finds its paths again in milliseconds rather than when iroh notices. The
relay voucher kept with the device key renews itself through the tunnel.

*Built, 2026-10-18, without the camera.*

- **Pairing by link.** The link from `kvad remote pair` is pasted, from
  Universal Clipboard, into the Server field. The phone pairs on the LAN or
  through the relay, using the link's voucher.
- **Every request through the tunnel**, with crates/tunnel's `for_ureq`
  connector: the CLI's code, moved into kvad-tunnel so that both use it.
- **The Keychain** (`security-framework`) holds the device key, and each
  server's key, voucher and device id, as one item that is "when unlocked,
  this device only", never synced.
- **The path** (`lan`, `direct`, `relay`) is shown in More and kept up to
  date.
- **The voucher renews itself** once half spent.
- **Unpair this phone** removes the device on the Mac and forgets it here.

Found on the way: an app whose executable cargo builds gets no
entitlements, on the simulator from the executable's own sections and on a
device from its signature. So `build.sh` links the simulator's sections into
cargo's binary and signs a device build with Xcode's entitlements file.
Without that, iOS refuses the Keychain with "a required entitlement isn't
present".

Tested in the iOS 26.5 simulator against a scratch kvad-serve:

- Paired over the LAN, reconnected by name from the Keychain, and chatted
  through the tunnel. The account is an administrator, and the device was
  shown acting as `user`.
- Then, with the server set to `direct = "off"` and a new Keychain, paired
  through `relay.kvad.eu`. More said `relay https://relay.kvad.eu`.
- Unpaired from the phone: the Mac closed its connection, and the app then
  says it is not paired.

*Added, 2026-10-18: links and the foreground.*

- **The Camera pairs.** The app registers the `kvad` URL scheme, so the
  system Camera, scanning the QR code from `kvad remote pair`, offers to open
  the link in Kvad; so does tapping one. The Slint app had no scanner of its
  own, since one needed an `objc2` binding to AVFoundation and the camera
  permission with it: it added the scene delegate's `scene:openURLContexts:`
  and `scene:willConnectToSession:options:` to Slint's
  `SlintWindowSceneDelegate` through the Objective-C runtime
  (`class_addMethod`), which implements neither. (The SwiftUI app has
  `.onOpenURL` and `scenePhase` instead, and now a scanner: below.) An opened link that parses as a server or a pairing link goes in
  the Server field with a line saying which server it is; it is never
  acted on until *Connect*. While connected, the chat says a link was opened.
- **Back in the foreground** (`sceneDidBecomeActive:`), the app tells the
  tunnel the network may have changed (kvad-tunnel's `network_changed`,
  iroh's `network_change`).

Tested in the simulator with `simctl openurl`: a link opened while the app
ran, and one that launched it, each filled the field; Connect then paired on
the LAN. A link opened while connected showed in the chat; a malformed one
was refused. The relay refused this scratch server's vouchers, since
`relay.kvad.eu`'s allow list no longer has its key, so that pairing was
on the LAN only.

*On a real iPhone, 2026-10-18.* An iPhone 14 Pro Max on iOS 27.0.1,
signed by a paid team, against a kvad-serve on the MacBook Pro at home
(`direct = "auto"`, `relay.kvad.eu`):

- The Camera read the QR code from `kvad remote pair` and opened the link
  in Kvad, which paired with *Connect*. The connection began on the relay
  and was on the LAN 4.5 s later.
- A chat on Wi-Fi: 32 tokens, the first in 23 ms.
- Wi-Fi off, in Settings: the same connection moved to the relay without
  dropping, 24 s after the chat.
- The app closed and opened again on mobile data: it reconnected from the
  Keychain through the relay. Two chats, 46 and 47 tokens, the first in
  26–29 ms, each reply 115 ms.
- No direct path on mobile data, as expected behind the carrier's NAT with
  no port opened on the home router: the relay is that path.

iOS gives an app the generic "iPhone" as the device's name without a
special entitlement, so that is the name the server shows.

Found on the way: Xcode's "Signing for Kvad requires a development team".
The team now goes in an uncommitted `ios/Team.local.xcconfig`, included by
`Signing.xcconfig`, which `xcodegen generate` leaves alone; a team chosen in
Xcode's window was thrown away by the next generate. The README has the
steps from a fresh Mac to the app on the phone.

*Added, 2026-10-19: a scanner of the app's own.* With SwiftUI the reason
for leaving it to the Camera was gone, so the Connect screen has *Scan a
pairing code* beside *Paste a pairing link* (`ios/Kvad/Scanner.swift`):

- **VisionKit's `DataScannerViewController`**, QR codes only, with no tap
  action on what it highlights. Where `isSupported` is false, AVFoundation's
  `AVCaptureMetadataOutput` with `.qr` alone. No third-party scanner.
- **The camera is asked for at the first Scan**, not at launch, with
  `NSCameraUsageDescription` saying it only reads the pairing code and that
  nothing is recorded or sent. Refused, the scanner says so and offers
  Settings, the Camera app and Paste. A device with no camera (the
  simulator) says that instead.
- **A code is only text for the core's `opened`**, the parser an opened link
  goes through. A code that parses as a server or pairing link fills in the
  Server field with the line saying which server it is for, the phone gives
  a success haptic, and the scanner closes; pairing still waits for
  *Connect*. Anything else is a note under the viewfinder, and the scanner
  looks on; each code goes to the core once, one at a time, while the
  scanner is open, so a second code in sight cannot replace the first. No
  code is opened as a URL, and nothing the camera sees is kept or sent.
- More has no way to add a server (a connected phone disconnects first), so
  Connect is the one place a pairing link is entered. The Camera's
  `kvad://` path and *Paste a pairing link* are as they were.

Tested in the iOS 26.5 simulator, which has no camera: the camera prompt
came at the first Scan ([asks](phone/scanner-asks.png)); refused, the
scanner said so with Open Settings ([refused](phone/scanner-refused.png));
allowed, it said there is no camera ([no camera](phone/scanner-no-camera.png)).
With codes read in by a debug build's `KVAD_PHONE_SCAN`, `https://example.com`
and `kvad://nothing` were each a note ([not Kvad](phone/scanner-not-kvad.png)),
and a pairing link closed the scanner, filled in the field and said which
server, with no connection tried ([read](phone/scanner-read.png)). A link
from `simctl openurl` and a pasted one went as before. A real scan is the
field check below.

Not yet done:

- *The scanner on a real iPhone*: the prompt at the first Scan, reading the
  QR code from `kvad remote pair` (the haptic, the field filled in, the
  scanner closed, pairing at *Connect*), a QR code that is not Kvad's (a
  note, the scanner open, nothing opened), and a refusal then Settings.
- *Pairing on mobile data alone*, through the relay with the link's voucher
  (done in the simulator, not yet from a phone).
- *A network change while the app stays in the background*, then back to
  the foreground (`sceneDidBecomeActive:`), without closing it.
- *A network change while in the foreground* (`NWPathMonitor`). iroh moved
  the connection on its own when Wi-Fi went off, within the 24 s above.

**Phase 5: the rest.** The More tab, screen by screen, ordered by what the
owner reaches for from a phone.

**Phase 6: on a real iPhone.** Signing with the owner's Apple developer
team, then TestFlight. A real phone on Wi-Fi and on mobile data is also
where `tools/relay-probe`'s field rows get their phone column: the Network
test under More (and on the Connect screen) is the probe's dialler, built
from its library, ahead of Phase 4 so that the run need not wait for it. Its
key is a file in the app's container until the Keychain question below is
settled.

## Open questions

- Is one binary for iPhone and iPad enough, or does the iPad want a sidebar
  layout of its own? SwiftUI's `NavigationSplitView` can do both from one
  app.
- Push notifications for long jobs (a video finished) need Apple's push
  service, which is a third party between the Mac and the phone. Is a local
  notification, when the app is next opened, enough?
