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

**Slint 1.18, Rust only.** Slint renders natively, through winit and Skia,
with no web view. Its UI is a declarative `.slint` markup compiled into the
binary, and everything else is Rust. Slint's iOS support covers what makes
an app feel at home: the safe area, the software keyboard's frame, Dynamic
Type's font size and the system's dark mode. It also takes part in UIKit's
scene lifecycle, which iOS 27 requires. Chosen by the owner over Dioxus,
whose iOS UI is a web view, and Tauri, whose UI would have been the Svelte
web UI.

Slint is licensed GPL-3.0, or under its royalty-free licence, which asks
for attribution: an "About Slint" mention in the app or its listing. The
app takes the royalty-free licence and says so in its About screen. Kvad
stays MIT.

**Its own workspace.** `apps/phone/` holds the app, with a `[workspace]` of
its own, as `tools/relay-probe` does. Slint and its renderer never enter
Kvad's builds or its CI time. Only the Xcode project is iOS-specific. The
Rust is not, so Android can follow later through Slint's Android backend.

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
  Cargo.toml          its own workspace; slint, ureq, serde_json
  build.rs            compiles ui/*.slint
  src/main.rs         starts the UI, wires callbacks to the client
  src/api.rs          the HTTP client: health, models, chat as a stream
  ui/app.slint        the window, the tab bar, the screens
  ui/theme.slint      colours and type, light and dark, after Apple's HIG
  ios/project.yml     XcodeGen: bundle, Info.plist, scene manifest
  ios/build.sh        Xcode's build phase: cargo for the right target
```

- **Threads.** The UI runs on the main thread, as UIKit requires. Each
  request runs on a worker thread, and its results go back with
  `slint::Weak::upgrade_in_event_loop`. No async runtime is needed until the
  tunnel brings iroh, and with it tokio.
- **The transport is pluggable.** `ureq` 3 takes its connection from a
  `Transport`. Plain TCP and TLS come first. The tunnel from
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
- **The phone's geometry.** Content inside `safe-area-insets`, the composer
  riding on top of the keyboard (`virtual-keyboard-position`), and touch
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

**Phase 2: Speak.** Recording with AVAudioSession through `cpal`, or a
small `objc2` binding if `cpal`'s iOS input falls short. 16 kHz mono, as
Whisper wants. Then the file endpoint, then the live session as the speech
plan lays it out, with words arriving as they are settled.

**Phase 3: Create, and model management.** Images and videos with job
progress over SSE. Loading and pulling models with their progress.

**Phase 4: the tunnel.** Pairing by QR code: the camera through a small
`objc2` binding to AVFoundation's code scanner. Then iroh as a `Transport`,
the keys in the Keychain, and the path shown in the connection's status
(`lan`, `direct`, `relay`).

**Phase 5: the rest.** The More tab, screen by screen, ordered by what the
owner reaches for from a phone.

**Phase 6: on a real iPhone.** Signing with the owner's Apple developer
team, then TestFlight. A real phone on Wi-Fi and on mobile data is also
where `tools/relay-probe`'s field rows get their phone column.

## Open questions

- Keychain access: through the `security-framework` crate, the usual one
  for Apple's Security framework from Rust, or a small `objc2` binding of
  our own. Phase 1 decides, by what it costs in the tree.
- Is one binary for iPhone and iPad enough, or does the iPad want a sidebar
  layout of its own? Slint can do both from one `.slint` tree.
- Push notifications for long jobs (a video finished) need Apple's push
  service, which is a third party between the Mac and the phone. Is a local
  notification, when the app is next opened, enough?
